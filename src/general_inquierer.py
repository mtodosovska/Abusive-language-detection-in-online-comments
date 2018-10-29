import pandas as pd
import re
import numpy as np

inq = pd.read_csv('../data/general inquierer/inquireraugmented.csv', encoding='latin1').iloc[1:]
inq = inq.drop('Source', axis=1).drop('Defined', axis=1).drop('Othrtags', axis=1)
inq['Entry'] = inq['Entry'].apply(lambda x: x.lower())
inq['Entry'] = inq['Entry'].apply(lambda x: re.sub(r'\#\w', '', x))
basic = pd.read_pickle('../features/basic_comments_tokenised.txt')

hvd = pd.DataFrame()
# lst = basic['comment'][0].split()
j = 0
for comm in basic['comment']:
    lst = comm.split()
    cat = []
    i = 0
    for s in lst:
        if s in inq['Entry'].tolist():
            # print(s, inq.iloc[i])

            cat.append([x for x in inq.iloc[i] if x != '0' and x != 0])
        i += 1

    cat = [item for sublist in cat for item in sublist]
    print('Comment:', j)
    terms = {}
    for c in cat:
        if c in terms.keys():
            terms[c] += 1
        else:
            terms[c] = 1

    row = []
    row.append(basic['rev_id'][j])
    j += 1
    for col in inq.columns[1:]:
        if col in terms.keys():
            row.append(terms[col])
        else:
            row.append(0)

    hvd = hvd.append(pd.DataFrame(row).transpose())

hvd.columns = np.append(['rev_id'], (inq.columns[1:].values))
hvd = hvd.reset_index(drop=True)
hvd.to_csv('../features/harvard4.csv', index=None)
print(hvd)
# inq['Entry'] ima duplikati