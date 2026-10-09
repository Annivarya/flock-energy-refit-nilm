.PHONY: data run test

data:
	python download_data.py
run:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python pipeline.py --stage all
	python write_reports.py
test:
	python -m unittest discover -s tests -v
