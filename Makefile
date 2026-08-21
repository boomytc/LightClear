# Makefile
.PHONY: help clean

.DEFAULT_GOAL := help

help: ## 显示帮助信息
	@echo "LightClear 根目录只做路由。先 cd 到 explore/light_clearvoice 或 products/<name>。"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean: ## 清理运行产物与缓存
	@echo "Cleaning..."
	@rm -rf logs output outputs
	@find . -path '*/.venv' -prune -o -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name '.pytest_cache' -exec rm -rf {} +
	@find . -type f -name '.DS_Store' -delete
	@echo "Cleaning completed!"
