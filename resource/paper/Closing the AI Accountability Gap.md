# Closing the AI Accountability Gap: Defining an End-to-End Framework for Internal Algorithmic

## 文章

- 提出了一个高层次内部算法审计框架（图 1）

- 外部审计和内部审计

  - 外部审计：项目完成部署后进行审计，如果产品有问题，不好的影响已经产生
  - 内部审计：贯穿在整个项目的开发过程中。在产品发布前能够及时纠正和弥补

- 人工智能已经有一些规范 （2.2)，但是行业如果把这个规范贯彻到产品里还没有实际方法

- 三个审计方面的传统行业案例分析

  - 航空：为什么飞行事故低，因为审计
  - 医学：诊断设备
  - 金融

- SMACTR 内部审计框架（图 2）

  - Scoping （技术参加）
  - Mapping
  - Artifact Collection（技术参加）
  - Testing
  - Reflection
  - Post-Audit
  - 每个阶段都会有审计的文档

- 假设了一家大型跨国软件公司 X，5 个原则

  - 透明度
  - 公正诚实无偏见
  - 安全和无害
  - 责任和问责
  - 隐私性

- 假设了 2 个客户

  - child abuse screening tool
  - Happy-Go-Lucky（提供了对应的 SMACTR 文档）

- 监管（属于 Other stakeholder）流程

  - anticipation
  - reflexivity
  - inclusion
  - responsiveness

- Scoping

  - 目标
  - 技术团队
    - 产品需求文档（PRD）或者项目 proposal
  - Artifact: Ethical Review of System Use Case
  - Artifact: Social Impact Assessment

- Mapping，辨识 stakeholders 以及 key collaborators

  - Artifact: Stakeholder Map
  - Artifact: Ethnographic Field Study

- Artifact Collection

  - Artifact: Design Checklist
  - Artifacts: Datasheets and Model Cards (技术团队)
    - Model cards: 描述模型性能特征的文档，可以对数据表进行补充
    - Datasheets: 数据的搜集过程

- Testing，对项目测试的阶段

  - Artifact: Adversarial Testing
  - Artifact: Ethical Risk Analysis Chart

- Reflection，对产品的决策和设计建议进行反思

  - Artifact: Algorithmic Use-related Risk Analysis 和 FMEA (Failure Model Effect Analysis)
  - Artifact: Remediation and Risk Mitigation Plan
  - Artifact: Algorithmic Design History File (ADHF)
  - Artifact: Algorithmic Audit Summary Report，这个总结报告把所有的审计要素，技术分析文档全部汇总，需要定量和定性

- Limitations of internal audits

## 结论

- 审计对 AI 的重要性（内部，外部方面）
- 内部审计和外部审计的不同
- 提出了内部审计的框架
- 技术团队在这个框架内应该如何提供和实现

- AI 识别有道德风险，隐私性
- 模型训练数据集的选择，需要考虑审计的要求 (e.g. no bias)
- 如果提供审计过程中需要的技术文档
