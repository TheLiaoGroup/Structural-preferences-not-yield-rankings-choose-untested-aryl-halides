"""Core estimates: molecule contributions; agreement across condition halves, partner halves and reactions;
held-out pair prediction; the second condition set; and the chemical readings of the contributions."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import argparse
import importlib
import itertools
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from public_common import *

def historical_contributions(w,c,p,A,G,connect_halves=False):
    """Both historical fitting stages, with published condition-rank parity."""
    full, halves, reliability, pair_tables = {},[{},{}],[],[]
    for rc, pairs in p.groupby("rxn_class",sort=True):
        rows=w[w.rxn_class==rc]
        order=c[c.rxn_class==rc].sort_values("coverage_rank").condition_id.tolist()
        def fit(selected):
            eff=A.two_way_fit(selected["yield"].values,selected.elec+" >> "+selected.nuc,selected.condition_id)
            key=pd.DataFrame([s.split(" >> ") for s in eff.index],columns=["elec","nuc"])
            table=key.assign(p=eff.values,rxn_class=rc)
            f=A.stage_two(table)
            return {"elec":f["a"],"nuc":f["b"]},table
        full[rc],table=fit(rows)
        pair_tables.append(table)
        for h in (0,1):
            selected=rows[rows.condition_id.isin(order[h::2])]
            if connect_halves:
                keep=G.DG.largest_piece(selected[["elec","nuc"]].drop_duplicates(),f"historical {rc} half {h}")
                selected=selected.merge(keep,on=["elec","nuc"])
            halves[h][rc],_=fit(selected)
        for side in ("elec","nuc"):
            a,b=halves[0][rc][side],halves[1][rc][side]
            shared=a.index.intersection(b.index)
            rho=G.spearman(a[shared],b[shared])
            reliability.append({"rxn_class":rc,"side":side,"n_molecules":len(shared),"r_half":rho,
                                "reliability":2*rho/(1+rho) if rho>-1 else np.nan})
    return full,pd.DataFrame(reliability),pd.concat(pair_tables,ignore_index=True)

def transfer_table(full,reliability,E,G,X):
    rel=reliability.set_index(["rxn_class","side"]).reliability
    records=[]
    for side in ("elec","nuc"):
        for ra,rb in itertools.combinations(sorted(full),2):
            a,b=full[ra][side],full[rb][side]
            shared=a.index.intersection(b.index)
            if len(shared)<5:continue
            va,vb=a[shared].values,b[shared].values
            ceiling=float(np.sqrt(max(rel[ra,side],0)*max(rel[rb,side],0)))
            rho=G.spearman(va,vb)
            ret,picked,best,mean=X.retention(va,vb) if len(shared)>=20 else (np.nan,)*4
            records.append({"side":side,"rxn_a":ra,"rxn_b":rb,"n_shared":len(shared),
                "same_reaction":frozenset((ra,rb)) in E.SAME_REACTION,"rho_obs":rho,"ceiling":ceiling,
                "rho_disattenuated":rho/ceiling if ceiling>0 else np.nan,"retention":ret,
                "B_mean_A_selected":picked,"B_mean_B_best":best,"B_mean_all":mean})
    return pd.DataFrame(records)

def pooled_point(table,column,G):
    t=table[~table.same_reaction]
    return G.weighted_mean(t[column],t.n_shared)

def export_effects(full,label,p,rows):
    out=[]
    for rc,sides in full.items():
        for side,values in sides.items():
            n=p[p.rxn_class==rc][side].value_counts()
            for mol,value in values.items():
                ids=rows[(rows.rxn_class==rc)&(rows[side]==mol)].source_well_id.tolist()
                out.append({"scale":"raw","label":label,"rxn_class":rc,"side":side,"mol":mol,
                    "main_effect":float(value),"n_pairs":int(n[mol]),"source_well_ids":";".join(ids)})
    return pd.DataFrame(out)

def comparison_membership(layouts,fits,k,G):
    records=[]
    for side,ra,rb in G.comparison_list():
        def selected(rc):
            lay=layouts[rc]
            names=lay.e_name if side=="elec" else lay.n_name
            return set(names[G.evaluable(lay,fits[rc],side,k)])
        shared=sorted(selected(ra)&selected(rb))
        if len(shared)<5:continue
        for mol in shared:
            values={f"{r}_{h}":float(fits[rc]["halves_s"][side][j][mol])
                for r,rc in (("a",ra),("b",rb)) for j,h in enumerate(("A","B"))}
            records.append({"side":side,"rxn_a":ra,"rxn_b":rb,"degree":k,"molecule":mol,**values})
    return pd.DataFrame(records)

def prediction_exports(lay,y,rows,G):
    records=[]; training_gamma=[]
    for f,(sub,test) in enumerate(lay.folds):
        got=G.fold_predictions(lay,y,sub,test)
        if got is None:raise RuntimeError(f"Failed fold: {lay.rxn}/{f}")
        test,target,pred,_=got
        _,gamma,stop=sub.fit(y)
        assert stop in (1,2)
        assert not set(sub.pairs)&set(test)
        for c,g in zip(lay.conds,gamma):
            training_gamma.append({"reaction":lay.rxn,"fold":f,"condition_id":c,"gamma_training":g,
                "training_pairs":len(sub.pairs),"training_wells":len(sub.wells)})
        for q,i in enumerate(test):
            e,n=lay.e_name[lay.pe[i]],lay.n_name[lay.pn[i]]
            idx=lay.pair_wells[i]
            used=idx[np.isfinite(gamma[lay.w_cond[idx]])]
            records.append({"reaction":lay.rxn,"pair_index":int(i),"pair_id":pair_id(lay.rxn,e,n),"fold":f,
                "elec":e,"nuc":n,"elec_degree":int(lay.deg["elec"][lay.pe[i]]),
                "nuc_degree":int(lay.deg["nuc"][lay.pn[i]]),"target":float(target[q]),
                "predicted_MEN":float(pred["MEN"][q]),"baseline_M0":float(pred["M0"][q]),
                "predicted_ME":float(pred["ME"][q]),"predicted_MN":float(pred["MN"][q]),
                "source_well_ids":";".join(rows.iloc[idx].source_well_id),
                "target_source_well_ids":";".join(rows.iloc[used].source_well_id),
                "target_wells":len(used),"training_pairs":len(sub.pairs),
                **{f"eligible_k{k}":bool(lay.deg['elec'][lay.pe[i]]>=k and lay.deg['nuc'][lay.pn[i]]>=k) for k in (2,3,5)}})
    return records,training_gamma

def run_points():
    wells,conditions,pairs=released()
    L,E,G=legacy()
    A=importlib.import_module("analysis_bba")
    X=importlib.import_module("extended_transfer_bba")
    S=importlib.import_module("gated_secondary_bba")
    sets={key:design_inputs(wells,conditions,pairs,key) for key in DESIGNS}
    w,c,p=sets["primary"]
    for label, (_, c0, p0) in sets.items():
        prefix=DESIGNS[label][4]
        write_csv(c0.rename(columns={"coverage_rank":"rank"}),RESULTS/f"{prefix}_conditions_bba.csv")
        write_csv(p0,RESULTS/f"{prefix}_pairs_bba.csv")
    deg=G.DG.degrees(p)
    eligible=G.DG.evaluable(p,deg)
    write_csv(G.DG.gate_b(p,deg,eligible),RESULTS/"gated_gateB_eligibility_bba.csv")
    layouts,ys,rows=layouts_from(w,c,p,G)
    fits={}; cells=[]; prediction=[]; half_records=[]; pair_predictions=[]; gammas=[]; design_rows=[]
    for rc,lay in layouts.items():
        print(f"Fit partner halves, pair folds and nominal intervals: {rc}",flush=True)
        fits[rc]=G.reaction_fits(lay,ys[rc])
        assert all(s in (1,2) for s in fits[rc]["stop"])
        cells.extend(G.gate_a_partner(lay,fits[rc]))
        prediction.extend(G.gate_a_prediction(lay,ys[rc]))
        rec,gs=prediction_exports(lay,ys[rc],rows[rc],G)
        pair_predictions.extend(rec);gammas.extend(gs)
        for side in ("elec","nuc"):
            names=lay.e_name if side=="elec" else lay.n_name
            for j,mol in enumerate(names):
                for h in (0,1):
                    value=fits[rc]["halves_s"][side][h].get(mol,np.nan)
                    half_records.append({"reaction":rc,"side":side,"half":"AB"[h],"molecule":mol,
                        "contribution":value,"degree":int(lay.deg[side][j]),
                        **{f"evaluable_k{k}":j in set(G.evaluable(lay,fits[rc],side,k)) for k in (2,3,5)}})
        design_rows.append({"rxn_class":rc,"conditions":len(lay.conds),"pairs":lay.n_pairs,
            "electrophiles":len(lay.e_name),"nucleophiles":len(lay.n_name),
            "median_partners_elec":float(np.median(lay.deg["elec"])),"median_partners_nuc":float(np.median(lay.deg["nuc"])),
            "in_extended_set":True,"wells":len(ys[rc]),"dense_decomposition":rc in L.REACTIONS})
    cells=pd.DataFrame(cells);prediction=pd.DataFrame(prediction);pp=pd.DataFrame(pair_predictions)
    write_csv(cells,RESULTS/"gated_A_cells_bba.csv")
    write_csv(prediction,RESULTS/"gated_A_prediction_bba.csv")
    write_csv(pd.DataFrame(half_records),RESULTS/"partner_half_contributions_bba.csv")
    write_csv(pp,RESULTS/"pair_holdout_predictions_bba.csv")
    write_csv(pd.DataFrame(gammas),RESULTS/"pair_holdout_training_condition_effects_bba.csv")
    figure2c=pp[(pp.reaction=="Acylation")&pp.eligible_k2].copy()
    figure2c.insert(1,"degree",2)
    write_csv(figure2c,RESULTS/"amide_heldout_pair_predictions.csv")
    write_csv(pd.DataFrame(design_rows),RESULTS/"design_counts.csv")
    write_csv(pd.DataFrame(design_rows),RESULTS/"extended_reactions_bba.csv")
    print("Gate A complete; recomputing full-data effects and retrospective selections",flush=True)
    full,rel,pe=historical_contributions(w,c,p,A,G)
    transfer=transfer_table(full,rel,E,G,X)
    write_csv(rel,RESULTS/"extended_reliability_bba.csv")
    write_csv(transfer,RESULTS/"extended_transfer_bba.csv")
    effects=export_effects(full,"primary",p,w)
    write_csv(effects,RESULTS/"full_data_effects_bba.csv")
    write_csv(effects,RESULTS/"molecules_bba.csv")
    write_csv(pe,RESULTS/"full_data_pair_effects_bba.csv")
    write_csv(pd.DataFrame([{"quantity":q,"estimate":pooled_point(transfer,q,G),"interval_status":"intervals in output/core_intervals/extended_pooled_bba.csv"}
                           for q in ("rho_obs","rho_disattenuated","retention")]),RESULTS/"extended_pooled_bba.csv")
    rng=np.random.default_rng(20261101)
    comparisons=[];loro=[]
    for k in (2,3,5):
        b=G.gate_b(layouts,fits,k,rng=rng)
        write_csv(b["observed"],RESULTS/f"gated_B_observed_k{k}_bba.csv")
        write_csv(b["relative"],RESULTS/f"gated_B_relative_k{k}_bba.csv")
        # Preserve old file schema for plots; explicit observed/relative tables remove its overlapping-key ambiguity.
        write_csv(pd.Series({**b["pooled observed"],**b["pooled relative"]}),RESULTS/f"gated_B_pooled_k{k}_bba.csv",index=True)
        write_json({"observed":b["pooled observed"],"relative":b["pooled relative"]},RESULTS/f"gated_B_pooled_k{k}_explicit.json")
        comparisons.append(comparison_membership(layouts,fits,k,G))
        R=b["relative"]
        for rc in sorted(set(R.rxn_a)|set(R.rxn_b)):
            remaining=R[(R.rxn_a!=rc)&(R.rxn_b!=rc)]
            loro.append({"degree":k,"omitted_reaction":rc,**G.relative_values(remaining)})
    write_csv(pd.concat(comparisons,ignore_index=True),RESULTS/"cross_reaction_partner_molecule_values_bba.csv")
    write_csv(pd.DataFrame(loro),RESULTS/"shortfall_leave_one_reaction_out.csv")
    secondary,_=S.gate_b_secondary(layouts,ys,cells)
    write_csv(secondary,RESULTS/"gated_B_secondary_bba.csv")
    print("Gate B and sensitivities complete; recomputing the second-condition-set check",flush=True)
    summaries={}
    for label,prefix in (("validation","holdout"),):
        wh,ch,ph=sets[label]
        fb=full
        fh,rh,peh=historical_contributions(wh,ch,ph,A,G)
        th=transfer_table(fh,rh,E,G,X)
        cs=[]
        for rc in sorted(fh):
            for side in ("elec","nuc"):
                a,b=fb[rc][side],fh[rc][side]
                shared=a.index.intersection(b.index)
                if len(shared)>=5:cs.append({"rxn_class":rc,"side":side,"n_shared":len(shared),"rho":G.spearman(a[shared],b[shared])})
        cs=pd.DataFrame(cs)
        summary=[{"quantity":"C1 within-reaction, analysed vs held-out","estimate":G.weighted_mean(cs.rho,cs.n_shared)},
                 {"quantity":"C2 cross-reaction, held-out, corrected","estimate":pooled_point(th,"rho_disattenuated",G)},
                 {"quantity":"E1 retention of a top-ten selection","estimate":pooled_point(th,"retention",G)}]
        summaries[prefix]=summary
        summary=pd.DataFrame(summary).assign(interval_status=f"intervals in output/core_intervals/{prefix}_summary_bba.csv",
            comparison_scope="molecular contributions fitted on two recorded condition sets")
        write_csv(summary,RESULTS/f"{prefix}_summary_bba.csv")
        write_csv(cs,RESULTS/f"{prefix}_cells_bba.csv")
        write_csv(th,RESULTS/f"{prefix}_transfer_bba.csv")
        write_csv(rh,RESULTS/f"{prefix}_reliability_bba.csv")
        write_csv(export_effects(fh,label,ph,wh),RESULTS/f"{prefix}_molecule_effects_bba.csv")
    print("Recomputing original six-reaction hypotheses (primary and logit scales)",flush=True)
    dense=w[w.rxn_class.isin(L.REACTIONS)]
    order=c[c.rxn_class.isin(L.REACTIONS)][["rxn_class","condition_id","coverage_rank"]].rename(columns={"coverage_rank":"rank"})
    original=[]
    for scale in ("raw","logit"):
        tables=A.run(dense,order,scale=scale,label="primary" if scale=="raw" else "logit sensitivity",inference=True)
        original.append(tables["results"])
        if scale == "raw":
            for name in ("pair_effects", "ceiling", "residual_spectrum"):
                write_csv(tables[name], RESULTS / f"{name}_bba.csv")
            write_csv(tables["molecules"], RESULTS / "dense_primary_molecules_bba.csv")
    write_csv(pd.concat(original,ignore_index=True),RESULTS/"results_bba.csv")

def main():
    run_points()
    import chemistry_public
    chemistry_public.main()
    import diagnose_transfer_bba
    wells, conditions, pairs = released()
    w, _, _ = design_inputs(wells, conditions, pairs, "primary")
    L, _, _ = load_modules()
    diagnose_transfer_bba.reproduce(w[w.rxn_class.isin(L.REACTIONS)])
    print("Core estimates and chemical readings complete",flush=True)

if __name__=="__main__":
    main()
