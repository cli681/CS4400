import pandas as pd
import numpy as np
from os.path import join
import Levenshtein as lev
from sklearn.ensemble import RandomForestClassifier

# 1. Data reading
ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))
print(ltable.count(),"\n\n",rtable.count())


# 2. Blocking
def block_by_brand(ltable, rtable):
    ltable['brand'] = ltable['brand'].astype(str)
    rtable['brand'] = rtable['brand'].astype(str)
    lbrands = set(ltable["brand"].values)
    brand2ids_l = {b.lower(): [] for b in lbrands}
    brand2ids_r = {b.lower(): [] for b in lbrands}
    for i, x in ltable.iterrows():
        brand2ids_l[x["brand"].lower()].append(x["id"])
    for i, x in rtable.iterrows():
        if x["brand"].lower() in brand2ids_r:
            brand2ids_r[x["brand"].lower()].append(x["id"])
    candset = []
    for brd in lbrands:
        l_ids = brand2ids_l[brd]
        r_ids = brand2ids_r[brd]
        for i in range(len(l_ids)):
            for j in range(len(r_ids)):
                candset.append([l_ids[i], r_ids[j]])
    return candset

def Extract_l(lst):
    return [item[0] for item in lst]
def Extract_r(lst):
    return [item[1] for item in lst]

def block_by_category(ltable, rtable, candset):
    ltable['category'] = ltable['category'].astype(str)
    rtable['category'] = rtable['category'].astype(str)
    lcat = set(ltable["category"].values)
    cat2ids_l = {c.lower(): [] for c in lcat}
    cat2ids_r = {c.lower(): [] for c in lcat}
    cand_l = Extract_l(candset)
    cand_r = Extract_r(candset)
    for i, x in ltable.iterrows():
        if x["id"] in cand_l:
            cat2ids_l[x["category"].lower()].append(x["id"])
    for i, x in rtable.iterrows():
        if ((x["category"].lower() in cat2ids_r) and (x["id"] in cand_r)):
            cat2ids_r[x["category"].lower()].append(x["id"])
    candset = []
    for brd in lcat:
        l_ids = cat2ids_l[brd]
        r_ids = cat2ids_r[brd]
        for i in range(len(l_ids)):
            for j in range(len(r_ids)):
                candset.append([l_ids[i], r_ids[j]])
    return candset

def combine(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR

candset = block_by_brand(ltable, rtable)
candset2 = block_by_category(ltable, rtable, candset)
print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
print("number of pairs after blocking",len(candset2))
candset_df = combine(ltable, rtable, candset2)

# 3. Feature engineering
def lev_jaro(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.jaro(x, y)

def lev_ratio(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.ratio(x, y)

def levenshtein_distance(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.distance(x, y)

def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "category", "brand", "modelno", "price"]
    features = []
    for attr in attrs:
        l_jaro = LR.apply(lev_jaro, attr=attr, axis=1)
        l_ratio = LR.apply(lev_ratio, attr=attr, axis=1)
        l_dist = LR.apply(levenshtein_distance, attr=attr, axis=1)
        features.append(l_jaro)
        features.append(l_ratio)
        features.append(l_dist)
    features = np.array(features).T
    return features


# 4. Model training and prediction
candset_features = feature_engineering(candset_df)
t_label = train.label.values
t_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
t_df = combine(ltable, rtable, t_pairs)
t_features = feature_engineering(t_df)
rf = RandomForestClassifier(n_estimators=2, max_depth=1, max_leaf_nodes=4, class_weight="balanced", random_state=0)
rf.fit(t_features, t_label)
y_pred = rf.predict(candset_features)


# 5. output
match_pairs = list(map(tuple, (candset_df.loc[y_pred == 1, ["id_l", "id_r"]]).values))
match_pairs_in_t = set(list(map(tuple, (t_df.loc[t_label == 1, ["id_l", "id_r"]]).values)))
pred_pairs = [p for p in match_pairs if p not in match_pairs_in_t]
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)