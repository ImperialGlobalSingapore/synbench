# Regenerate the paper's figures and tables from results/ (minutes, CPU only).
PY ?= python

.PHONY: paper figures tables clean
paper: figures tables

figures:
	cd paper && $(PY) fig2_leakage.py && $(PY) fig3_mia_roc.py && $(PY) fig4_size_lora.py

tables:
	cd paper && $(PY) tab23_utility.py && $(PY) tab4_fidelity.py && $(PY) tab5_size.py && $(PY) tab5_size.py --stat best

# Rebuild the leakage summary from the per-document Infini-gram results.
results/leakage/redpajama_summary.csv: $(wildcard results/leakage/redpajama/*.jsonl)
	$(PY) src/leakage/summarize_leakage.py

clean:
	rm -rf paper/figs paper/tables
