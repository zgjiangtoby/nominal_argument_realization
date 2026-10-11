"""Read keyed entity scores and fit grouped prediction models."""

import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def read_probes(directory,score_path):
    score_rows = read_jsonl(score_path)
    scores = {r["id"]:r for r in score_rows}
    assert len(scores) == len(score_rows),"Duplicate score IDs"
    groups = defaultdict(list)
    seen = set()
    for line in (directory/"inputs.jsonl").open():
        source = json.loads(line)
        key = source["id"]
        assert key not in seen and key in scores,key
        seen.add(key)
        score = scores[key]
        assert score["labels"] == source["labels"]
        logp = np.asarray(score["answer_log_probabilities"],dtype=float)
        assert len(logp) == len(source["entity_codes"]) and np.isfinite(logp).all()
        p = np.exp(logp-logsumexp(logp))
        groups[(source["item_id"],source.get("role","subject"),source["condition"])].append(
            (source["order"],dict(zip(source["entity_codes"],p)),score["answer_probability_mass"]))
    assert seen == set(scores),"Scores must cover every input prompt exactly once"
    result = {}
    for key,group in groups.items():
        assert sorted(r[0] for r in group) == [0,1]
        a,b = group[0][1],group[1][1]
        assert set(a) == set(b) and "OTHER" in a
        averaged = {entity:0.5*(a[entity]+b[entity]) for entity in a}
        total = sum(value for entity,value in averaged.items() if entity != "OTHER")
        assert total > 0
        conditional = {entity:value/total for entity,value in averaged.items() if entity != "OTHER"}
        n = len(conditional)
        entropy = -sum(p*np.log(p) for p in conditional.values() if p>0)
        result[key] = {"probabilities":conditional,"entropy":entropy/np.log(n) if n>1 else 0.,
                       "other_mass":averaged["OTHER"],"answer_mass":0.5*(group[0][2]+group[1][2]),
                       "order_total_variation":0.5*sum(abs(a[k]-b[k]) for k in a),
                       "order_top_disagreement":int(max(a,key=a.get) != max(b,key=b.get))}
    return result


def gold_features(probe,gold):
    probability = probe["probabilities"][gold]
    return {"surprisal":float(-np.log(max(probability,1e-300))),
            "correct":int(max(probe["probabilities"],key=probe["probabilities"].get)==gold),
            "probability":float(probability),
            **{key:probe[key] for key in ("entropy","other_mass","answer_mass","order_total_variation","order_top_disagreement")}}


def fit_predict(x,y,weights,train,test,expanded_training_rows=None):
    assert len(np.unique(y[train])) == 2 and weights[train].sum() > 0
    w = weights[train]
    expanded_training_rows = int(train.sum()) if expanded_training_rows is None else float(expanded_training_rows)
    w = w/w.sum()*expanded_training_rows
    scale = StandardScaler(with_mean=False).fit(x[train],sample_weight=w)
    a = scale.transform(x[train])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always",ConvergenceWarning)
        model = LogisticRegression(C=1.,solver="lbfgs",tol=1e-9,max_iter=2000).fit(a,y[train],sample_weight=w)
    failures = [str(message.message) for message in captured if issubclass(message.category, ConvergenceWarning)]
    for message in captured:
        if not issubclass(message.category, ConvergenceWarning):
            warnings.warn(message.message, message.category)
    fallback = None
    solver = "lbfgs"
    if failures:
        original_eta = model.decision_function(a)
        original_gradient = np.r_[a.T@(w*(expit(original_eta)-y[train]))+model.coef_[0],
                                   (w*(expit(original_eta)-y[train])).sum()]
        original_objective = float((w@(np.logaddexp(0,original_eta)-y[train]*original_eta)+.5*np.dot(model.coef_[0],model.coef_[0]))/w.sum())
        original_prediction = model.predict_proba(scale.transform(x[test]))[:,1]
        fallback = {"reason":failures,"original_iterations":int(model.n_iter_[0]),
                    "original_gradient_max_per_training_weight":float(np.abs(original_gradient).max()/w.sum())}
        with warnings.catch_warnings():
            # sklearn may warn about its Hessian and silently switch back to L-BFGS.
            # Any fallback warning is fatal, including scipy.linalg.LinAlgWarning.
            warnings.simplefilter("error")
            model = LogisticRegression(C=1.,solver="newton-cholesky",tol=1e-9,max_iter=2000).fit(a,y[train],sample_weight=w)
        solver = "newton-cholesky"
        eta = model.decision_function(a)
        objective = float((w@(np.logaddexp(0,eta)-y[train]*eta)+.5*np.dot(model.coef_[0],model.coef_[0]))/w.sum())
        assert objective <= original_objective+1e-10, (original_objective,objective)
        fallback.update(objective_change_per_training_weight=objective-original_objective,
                        max_prediction_difference_from_warning_fit=float(np.max(np.abs(model.predict_proba(scale.transform(x[test]))[:,1]-original_prediction))))
    residual = w*(expit(model.decision_function(a))-y[train])
    gradient = np.r_[a.T@residual+model.coef_[0],residual.sum()]
    record = {"iterations":int(model.n_iter_[0]),"solver":solver,"fallback":fallback,
              "gradient_max_per_training_weight":float(np.abs(gradient).max()/w.sum()),
              "intercept":float(model.intercept_[0]),"coefficients_original_units":(model.coef_[0]/scale.scale_).tolist(),
              "training_scales":scale.scale_.tolist()}
    assert record["gradient_max_per_training_weight"] < 1e-5,record
    return model.predict_proba(scale.transform(x[test]))[:,1],record


def binary_losses(y,p):
    p = np.clip(p,1e-12,1-1e-12)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def grouped_evaluation(frame,columns,group_key,fold_key,equal_groups,refits,seed,coefficient_bootstrap=False):
    y = frame.y.to_numpy(dtype=int)
    groups,group_indices = np.unique(frame[group_key].astype(str),return_inverse=True)
    folds = frame[fold_key].to_numpy(dtype=int)
    assert frame.groupby(group_key)[fold_key].nunique().eq(1).all()
    n_groups = len(groups)
    counts = np.bincount(group_indices)
    base_weights = 1/counts[group_indices] if equal_groups else np.ones(len(frame))
    matrices = {name:frame[names].to_numpy(dtype=float) for name,names in columns.items()}
    assert all(np.isfinite(x).all() for x in matrices.values())
    predictions = {name:np.full(len(frame),np.nan) for name in columns}
    fits = []
    for fold in sorted(set(folds)):
        test,train = folds==fold,folds!=fold
        for name,x in matrices.items():
            p,record = fit_predict(x,y,base_weights,train,test)
            predictions[name][test] = p
            fits.append({"model":name,"fold":int(fold),**record})
    assert all(np.isfinite(p).all() for p in predictions.values())
    loss = {name:binary_losses(y,p) for name,p in predictions.items()}
    score = {name:float(np.average(v,weights=base_weights)) for name,v in loss.items()}
    full_fits = {}
    for name,x in matrices.items():
        _,record = fit_predict(x,y,base_weights,np.ones(len(y),dtype=bool),np.ones(len(y),dtype=bool))
        full_fits[name] = {"feature_names":columns[name],**record}
    rng = np.random.default_rng(seed)
    fixed = {name:[] for name in columns}
    refitted = {name:[] for name in columns}
    coefficient_draws = {name:[] for name in columns}
    numerics = {"refit_fits":0,"maximum_refit_gradient_per_training_weight":0.,"refit_solver_fallbacks":[]}

    def record_refit(record, name, draw, fold, phase):
        numerics["refit_fits"] += 1
        numerics["maximum_refit_gradient_per_training_weight"] = max(numerics["maximum_refit_gradient_per_training_weight"],record["gradient_max_per_training_weight"])
        if record["fallback"] is not None:
            numerics["refit_solver_fallbacks"].append({"model":name,"draw":draw,"fold":fold,"phase":phase,**record})
    for draw in range(max(2000,refits)):
        cluster_weights = rng.multinomial(n_groups,np.full(n_groups,1/n_groups))
        weights = base_weights*cluster_weights[group_indices]
        for name in columns:
            fixed[name].append(float(np.average(loss[name],weights=weights)))
        if draw < refits:
            for name,x in matrices.items():
                p = np.full(len(frame),np.nan)
                for fold in sorted(set(folds)):
                    train = (folds!=fold)&(weights>0)
                    test = (folds==fold)&(weights>0)
                    if not test.any():
                        continue
                    expanded_rows = cluster_weights[group_indices[train]].sum()
                    predicted,record = fit_predict(x,y,weights,train,test,expanded_rows)
                    record_refit(record,name,draw,int(fold),"heldout")
                    p[test] = predicted
                present = weights>0
                assert np.isfinite(p[present]).all()
                refitted[name].append(float(np.average(binary_losses(y[present],p[present]),weights=weights[present])))
                if coefficient_bootstrap:
                    _,record = fit_predict(x,y,weights,present,np.ones(len(y),dtype=bool),cluster_weights[group_indices[present]].sum())
                    record_refit(record,name,draw,None,"full_coefficient")
                    coefficient_draws[name].append([record["intercept"],*record["coefficients_original_units"]])
    details = {}
    for name,p in predictions.items():
        details[name] = {"nll":score[name],"brier":float(np.average((p-y)**2,weights=base_weights)),
                         "accuracy":float(np.average((p>=.5)==y,weights=base_weights))}
    return {"predictions":predictions,"fits":fits,"full_fits":full_fits,"metrics":details,
            "fixed_draws":{k:np.asarray(v) for k,v in fixed.items()},
            "refit_draws":{k:np.asarray(v) for k,v in refitted.items()},
            "coefficient_draws":{k:np.asarray(v) for k,v in coefficient_draws.items()},
            "groups":n_groups,"rows":len(y),"weighting":"group_equal" if equal_groups else "row_equal","numerics":numerics}


def contrast(result,new,old):
    delta = result["metrics"][new]["nll"]-result["metrics"][old]["nll"]
    fixed = result["fixed_draws"][new]-result["fixed_draws"][old]
    refit = result["refit_draws"][new]-result["refit_draws"][old]
    chosen = refit if len(refit)>1 else fixed
    se = float(np.std(chosen,ddof=1))
    return {"comparison":f"{new}-{old}","delta_nll":delta,
            "fixed_prediction_95_interval":np.quantile(fixed,[.025,.975]).tolist(),
            "paired_refit_95_interval":np.quantile(refit,[.025,.975]).tolist() if len(refit) else None,
            "refit_draws":len(refit),"bootstrap_standard_error":se,
            "normal_approximation_p":float(2*norm.sf(abs(delta)/se)) if se>0 else None,
            "conditional_on_frozen_partitions":True}


def holm(values):
    values = np.asarray(values,dtype=float)
    assert values.ndim==1 and len(values)>0 and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(), "Holm requires finite, valid p-values for the declared family"
    order = np.argsort(values)
    adjusted = np.minimum(1,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    result = np.empty(len(values))
    result[order] = adjusted
    return result.tolist()


def cluster_summary(frame,columns,group_key,equal_groups,seed=20261009,draws=2000):
    """Paired fixed-measurement means under whole-group reweighting."""
    sums = frame.groupby(group_key)[columns].sum()
    counts = frame.groupby(group_key).size().reindex(sums.index).to_numpy(dtype=float)
    matrix = sums.to_numpy(dtype=float)
    assert len(sums)>1 and np.isfinite(matrix).all()
    if equal_groups:
        matrix /= counts[:,None]
        counts = np.ones(len(sums))
    point = matrix.sum(0)/counts.sum()
    rng = np.random.default_rng(seed)
    values = []
    for offset in range(0,draws,100):
        multiplicity = rng.multinomial(len(sums),np.full(len(sums),1/len(sums)),size=min(100,draws-offset))
        values.append((multiplicity@matrix)/(multiplicity@counts)[:,None])
    bootstrap = np.concatenate(values)
    return {name:{"mean":float(point[j]),"fixed_measurement_95_interval":np.quantile(bootstrap[:,j],[.025,.975]).tolist(),
                  "bootstrap_se":float(bootstrap[:,j].std(ddof=1)),"groups":len(sums),"rows":len(frame),
                  "weighting":"group_equal" if equal_groups else "row_equal","draws":draws}
            for j,name in enumerate(columns)}


