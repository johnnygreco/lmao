.PHONY: docs


clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache


docs:
	cp README.md docs/index.md
	python -m mkdocs serve


deploy-docs:
	cp README.md docs/index.md
	python -m mkdocs gh-deploy


prerelease: 
	semantic-release --prerelease publish
