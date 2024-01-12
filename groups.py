
import numpy as np
import pandas as pd
import os.path as op
import re
import copy
import warnings

from collections.abc import Sequence, Hashable, Callable, Iterable
from difflib import SequenceMatcher

# NOTE: make sure your phrasing of the five options for programming skills satisfy these
# patterns, OR extend the patterns to match your phrasing.
_default_skill_replacement = {
    re.compile(r'(.*[Hh]ardly.*)|(.*bad.*)'):                               1,
    re.compile(r'(.*[Bb]arely.*)|(.*so-so.*)|(.*here and there.*)'):        2,
    re.compile(r'.*average.*'):                                             3,
    re.compile(r'(.*enjoy.*)|(.*good.*)'):                                  4,
    re.compile(r'(.*wizard.*)|(.*excellent.*)'):                            5
}

# NOTE: make sure your phrasing of the five options for domain expertise levels satisfy these
# patterns, OR extend the patterns to match your phrasing.
_default_application_domain_expertise_replacement = {
    re.compile(r'(.*don\'t have.*)|(.*informatics.*)'):     1,
    re.compile(r'(.*physics.*)|(.*chemistry.*)'):           2,
    re.compile(r'(.*dabbled.*)|(.*free time.*)'):           3,
    re.compile(r'(.*identify myself.*)'):                   4,
    re.compile(r'(.*domain specialist.*)'):                 5
}

### HELPER FUNCTIONS

def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def interests_compatibility(df, studix1: int, studix2: int, col_interests: str) -> float:
    s1 = str(df.loc[studix1, col_interests]).lower().strip()
    s2 = str(df.loc[studix2, col_interests]).lower().strip()

    if 'nan' in (s1, s2):
        return 0.25  # compatible but not a great overlap (better than totally different interests; but not great [=1.0])
    else:
        return text_similarity(s1, s2)
    

# helper function: I could not get panda's replace() function to work for some reason...
def replace(df, colname, to_replace: dict, full_replace=True, only_repl_str=True) -> None:
    for rowix in range(len(df)):
        if not only_repl_str or isinstance(df.loc[rowix, colname], str):
            row_replaced = False
            for regexp, val in to_replace.items():
                if not regexp.match(df[colname][rowix]) is None:
                    df.loc[rowix, colname] = val
                    row_replaced = True

                    break
            
            if full_replace:
                assert row_replaced, f'for row value "{df[colname].iloc[rowix]}" I could not find a replacement rule.'
        else:
            pass  # item was not a string and did not have to be processed

    return


### DATA READING/WRITING FUNCTIONS:


def read_student_records(path: str, cols_skills: Sequence[str], col_email='Email', col_name='Name',
                         col_background=None, col_interests=None, sheet_name=0, to_replace: Sequence[dict] = None, verbose=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)

    if verbose > 0:
        print(f'read_student_records: the original data contains {len(df)} rows and {len(df.columns)} columns.')

    columns = list(cols_skills)
    columns.append(col_email)
    columns.append(col_name)
    if not col_background is None:
        columns.append(col_background)
    if not col_interests is None:
        columns.append(col_interests)

    if not columns is None:
        df = df[columns]
    
    if not to_replace is None:
        assert len(to_replace) == len(cols_skills), 'I expect a replacement dict (or None) for every element of `cols_skills`.'

        for ix, repl_rule in enumerate(to_replace):
            if not repl_rule is None:
                assert isinstance(repl_rule, dict), f'{repl_rule=} was expected to be a dict'

                replace(df, cols_skills[ix], repl_rule)  # in-place
    # df.replace(to_replace=to_replace, inplace=True)  # for some reason this does not work... replaced by above

    # df.astype({colname: 'float' for colname in cols_skills}, copy=False)  # does not work either... sigh
    df = df.astype({colname: 'float' for colname in cols_skills}, copy=True)

    if verbose > 1:
        print(f'read_student_records: before dropping NaN, the data contains {len(df)} rows.')

    # clean records (missing values etc.)
    df.dropna(subset=[col_name, col_email] + cols_skills, inplace=True)

    if verbose > 0:
        print(f'read_student_records: after dropping NaN, the data contains {len(df)} rows.')

    df.drop(df[df[col_email] == 'anonymous'].index, inplace = True)

    if verbose > 0:
        print(f'read_student_records: after dropping email=="anonymous", the data contains {len(df)} rows.')
    
    return df


### SCHEDULING FUNCTIONS:


def infer_group_size_array(n: int, avg: int):
    assert n >= avg, 'how can the expected average group size be smaller than the total size? Intended?'

    max_num_full_groups = int(np.floor(n / avg))
    remainder = n % avg

    if remainder > 0:
        group_sizes = [avg]*max_num_full_groups + [remainder]

        while group_sizes[-1] < avg - 1:
            # use the current value as index to ensure stealing from different groups each iteration:
            ix_to_steal_from = avg - 2 - group_sizes[-1]

            assert 0 <= ix_to_steal_from
            assert ix_to_steal_from < len(group_sizes) - 1, 'should not steal from itself'

            group_sizes[ix_to_steal_from] -= 1
            group_sizes[-1] += 1
    else:
        group_sizes = [avg]*max_num_full_groups
    
    return group_sizes


_fitness_over_epochs = []
def optimize_assignment_for_diversity(df: pd.DataFrame, cols_skills: Sequence[str], group_sizes: Sequence[int] | int, 
                                      initial_groups=None, max_contig_no_improvements=50, max_iter=2000, steepness=(100., 30.), 
                                      interests_weight=0.5, col_interests=None, verbose=0):
    global _fitness_over_epochs

    if np.isscalar(group_sizes):
        group_sizes = infer_group_size_array(len(df), group_sizes)

    assert sum(group_sizes) == len(df), 'every student is expected to be in a group'

    if initial_groups is None:
        student_ixs = list(np.random.permutation(df.index))

        groups = []

        for gix in range(len(group_sizes)):
            stud_ixs_g = [student_ixs.pop() for _ in range(group_sizes[gix])]
            groups.append(stud_ixs_g)
    else:
        groups = copy.deepcopy(initial_groups)
    
    assert len(groups) == len(group_sizes)

    def skillvec(stud_ix):
        return df.loc[stud_ix, cols_skills].to_numpy().astype(float)
    
    def diversity(stud_ix1, stud_ix2):
        return np.linalg.norm(skillvec(stud_ix1) - skillvec(stud_ix2))

    def fitness_assignment(groups, interests_weight=interests_weight, col_interests=col_interests):
        diversities = [[diversity(six1, six2) 
                        for six1 in stud_ixs_g for six2 in stud_ixs_g if six1 > six2] 
                        for stud_ixs_g in groups]
        
        fitness = np.array(list(map(np.mean, diversities)), dtype=float)

        if interests_weight > 0.0 and not col_interests is None:
            interests_overlaps = [[interests_compatibility(df, six1, six2, col_interests) 
                                   for six1 in stud_ixs_g for six2 in stud_ixs_g if six1 > six2] 
                                   for stud_ixs_g in groups]
            
            avg_overlap_per_group = np.array(list(map(np.mean, interests_overlaps)), dtype=float)

            assert np.all(0.0 <= avg_overlap_per_group), 'interests overlap should be between 0 and 1'
            assert np.all(avg_overlap_per_group <= 1.0), 'interests overlap should be between 0 and 1'

            avg_overlap_per_group *= interests_weight

            assert len(avg_overlap_per_group) == len(fitness)

            # inflate each group's fitness by a factor that scales with the interests overlap
            # note: in a 2023 cohort of the CSS course, the average interests_compatibility between
            # students was about 0.25.
            try:
                return np.mean(np.product([fitness, np.add(1.0, avg_overlap_per_group)], axis=0))
            except TypeError as e:
                print(f'error: {fitness=}, {np.add(1.0, avg_overlap_per_group)=}')

                raise TypeError(e)
        else:
            return np.mean(fitness)
    
    def step(cur_fit: float, groups: Sequence[int], steepness=steepness):
        gix1, gix2 = np.random.choice(range(len(groups)), 2, replace=False)
        g1_six = np.random.choice(range(len(groups[gix1])))
        g2_six = np.random.choice(range(len(groups[gix2])))

        # swap
        interm = groups[gix1][g1_six]
        groups[gix1][g1_six] = groups[gix2][g2_six]
        groups[gix2][g2_six] = interm

        new_fit = fitness_assignment(groups)

        if new_fit > cur_fit:
            return new_fit  # done
        else:
            p = np.exp(steepness * (new_fit - cur_fit)) * 0.5

            if np.random.uniform() >= p:
                # swap back
                interm = groups[gix1][g1_six]
                groups[gix1][g1_six] = groups[gix2][g2_six]
                groups[gix2][g2_six] = interm

                return cur_fit
            else:
                return new_fit

    cur_fit = fitness_assignment(groups)

    _fitness_over_epochs = [cur_fit]  # store initial fitness value
    
    # make a cur_steepness function which is used in the loop below to determine the correct steepness parameter for optimization
    # (inverse temperature basically)
    if np.isscalar(steepness):
        cur_steepness = lambda i, n: steepness
    else:
        assert len(steepness) == 2, f'steepness expected in the format of (high, low), not {steepness=}'
        if not steepness[0] >= steepness[1]:
            warnings.warn('steepness expected in the format of (high, low) with high >= low. Did you really intend to make a temperature gradient that goes up?')

        cur_steepness = lambda i, n: steepness[0] - float(i)/(n - 1) * (steepness[0] - steepness[1])

    best_fit = cur_fit
    best_groups = copy.deepcopy(groups)  # keep track of best solution
    num_contig_no_improvement = 0
    for trial in range(max_iter):
        new_fit = step(cur_fit, groups, steepness=cur_steepness(trial, max_iter))

        if new_fit <= cur_fit:
            num_contig_no_improvement += 1
        else:
            num_contig_no_improvement = 0

            best_fit = cur_fit
            best_groups = copy.deepcopy(groups)  # keep track of best solution
        
        cur_fit = new_fit  # in case we will continue

        _fitness_over_epochs.append(cur_fit)  # store fitness value
        
        if num_contig_no_improvement >= max_contig_no_improvements:
            break
    
    return cur_fit, groups


if __name__ == "__main__":
    pass