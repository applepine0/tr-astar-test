# 项目结构
```text
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
```
# 命名规范推荐
1. 基础格式规范（PEP 8 标准）
变量、函数、方法：全小写，单词间用下划线分隔。snake_case

✅ vertices, expand, is_near_zero, calculate_distance

类名、异常：首字母大写（大驼峰）。PascalCase

✅ Polygon, Pose2D, CollisionError

全局常量：全大写，下划线分隔。UPPER_SNAKE_CASE

✅ MAX_SPEED, GEOMETRY_EPSILON, PI

模块（文件）名：全小写，尽量简短。

✅ geometry.py, pose_utils.py

2. 在几何、机器人和物理引擎中，仅有基础格式是不够的，极易造成单位混淆或坐标系混乱。

① 单位后缀法（避免火星探测器坠毁惨案）
永远在变量名中带上单位。

✅ x_m （x坐标，单位米）

✅ theta_rad （角度，弧度制） / theta_deg （角度制）

✅ timeout_ms （超时时间，毫秒）

✅ length_px （像素长度）

反例：x, angle, timeout （极其容易产生歧义）。

② 坐标系来源-目标后缀法（Frame Suffix）
在机器人和 SLAM 中，必须标明这个位姿/向量是在哪个坐标系下。

✅ pose_world （世界坐标系下的位姿）

✅ pose_base （机器人基座坐标系下的位姿）

✅ T_world_base （从基座坐标系到世界坐标系的变换矩阵）

反例：current_pose, target_pose （不知道是相对于谁的）。

③ 布尔值前缀法
必须让人一眼看出这是 True/False，通常用 is_, has_, can_, should_ 开头。

✅ is_closed （多边形是否闭合）

✅ is_convex （是否为凸多边形）

✅ has_collision （是否发生碰撞）

✅ can_expand （是否可以膨胀）

反例：status, flag, check （这种词你无法判断真假值的语义）。

④ 差值/增量前缀法
在计算位移、旋转时，dx, dy, d_theta 是数学惯例。但在代码中，推荐写全 delta 或使用明确的前缀。

✅ delta_x, delta_y 或 dx, dy（限于局部极短的数学循环）

✅ prev_vertex / curr_vertex （上一个/当前顶点）

✅ min_x / max_x