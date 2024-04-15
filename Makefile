lint:
	poetry run black .
	poetry run isort .
	poetry run ruff check .
	poetry run mypy .


make_requirement_dir:
	mkdir -p requirements


gen_requirement: make_requirement_dir
	poetry export -f requirements.txt --output requirements/requirements.txt

	
gen_dev_requirement: make_requirement_dir
	poetry export -f requirements.txt --output requirements/dev-requirements.txt --with optional,dev,test


gen_requirements: gen_requirement gen_dev_requirement
