# 项目结构

tr-planning/
├── pyproject.toml
├── uv.lock                  # 提交到 git
├── .python-version
├── src/
│   └── tr_planning/
│       ├── core/            # 最底层：纯数据与几何，无算法依赖
│       │   ├── params.py    # 底盘参数：外廓/轮径/v_max/δ̇max/舵角限位
│       │   ├── geometry.py  # 点、多边形、位姿
│       │   └── field.py     # 走廊、L1 立面、围栏、隔板、基座
│       ├── planning/        # 搜索相关
│       │   ├── occupancy.py # 占据图 + 规则外扩
│       │   ├── cost.py      # 时间下界（max 组合）
│       │   └── astar.py     # 二维/三维 A*
│       ├── traj/            # 轨迹相关
│       │   ├── postproc.py  # 拉直 / 平滑 / 朝向调度
│       │   └── topp.py      # 包装 toppra
│       └── io/
│           └── ref_frame.py # 以后：给主控的参考帧
├── configs/
│   └── field_demo.toml      # 场地与机器人参数（不硬编码）
├── scripts/                 # 一次性实验脚本，不进包
│   ├── demo_path.py
│   └── plot_field.py
└── tests/
    ├── test_cost_admissible.py
    └── test_astar_basic.py