def get_talley(adjective_pairs):
    adjective_pairs['combine'] = [sorted(x) for x in (adjective_pairs['adj_premise'] + ' ' + adjective_pairs['adj_hypothesis']).str.split()]
    talley = adjective_pairs['combine'].value_counts()
    return talley