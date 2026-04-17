---
name: ultralytics modification policy
description: 禁止修改 ultralytics 文件夹，它是官方子模块
type: feedback
---

ultralytics 文件夹是从官方拉取的子模块，禁止修改。尽量使用原本的实现，原本的作为后端，项目只做前端封装。

**Why:** ultralytics 是官方 git 子模块，修改后后续 `git pull` 更新会很麻烦，容易产生冲突或丢失修改。

**How to apply:**
1. 优先使用 ultralytics 原有功能，只在前端做封装调用
2. 确实需要修改时，在项目根目录创建对应结构的文件，例如需要改 `ultralytics/utils/downloads.py`，则创建 `utils/downloads.py`，然后修改导入路径
3. 现有的 `utils/downloads.py` 就是这个模式，用于将模型下载路径重定向到项目的 weights 目录
