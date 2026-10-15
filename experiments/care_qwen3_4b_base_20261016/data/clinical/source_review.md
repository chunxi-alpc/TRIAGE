# 临床规则来源与核验说明

真实临床规则的忠实转述，模拟患者，无真实患者数据。核验由助手基于公开一手参考资料完成，未声称经过独立医生复核。
临床阈值、人群、单位、边界和必要例外已核对；模型概率不参与金标产生。不以高置信度作为资格条件。

- **成人BMI**：[公开一手来源](https://fingertips.phe.org.uk/profile/obesity-physical-activity-nutrition/supporting-information/definitions)；依据：Definition of adult BMI categories。
- **贫血程度**：[公开一手来源](https://www.ncbi.nlm.nih.gov/books/NBK602185/table/ch5.tab3/)；依据：WHO 2024 Table 3, adults 15-65 nonpregnant women。
- **房颤抗凝推荐措辞**：[公开一手来源](https://www.nice.org.uk/guidance/NG196/chapter/recommendations)；依据：Recommendations 1.6.3, 1.6.4 and 1.6.7。
- **血压目标**：[公开一手来源](https://www.nice.org.uk/guidance/ng136/chapter/recommendations)；依据：Recommendations 1.4.20-1.4.21；Recommendation 1.4.22。
- **DXA T-score范围**：[公开一手来源](https://www.niams.nih.gov/health-topics/bone-mineral-density-tests-what-numbers-mean)；依据：What does a T-score mean?。
- **COPD气流受限程度**：[公开一手来源](https://www.nice.org.uk/guidance/ng115/chapter/Recommendations)；依据：Recommendation 1.1.26, Table 4, NICE CG101 column。
- **血糖分类**：[公开一手来源](https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis)；依据：Test results for diagnosis of prediabetes and diabetes；Test results table, fasting plasma glucose。
- **低血糖分级**：[公开一手来源](https://doi.org/10.2337/dc26-s006)；依据：ADA Standards of Care 2026, Table 6.4。
- **肺炎风险评分**：[公开一手来源](https://www.nice.org.uk/guidance/NG250/chapter/recommendations)；依据：Box 1, CRB65 mortality risk in primary care；Box 2, CURB65 mortality risk in hospital。
- **CKD GFR分类**：[公开一手来源](https://www.kidney.org/why-classify-ckd)；依据：GFR categories G1-G5。
- **咽痛抗菌药推荐措辞**：[公开一手来源](https://www.nice.org.uk/guidance/ng84/chapter/recommendations)；依据：Recommendations 1.1.6 and 1.1.10；Recommendations 1.1.6, 1.1.8 and 1.1.10。
- **Wells概率分类**：[公开一手来源](https://www.nice.org.uk/guidance/ng158/chapter/recommendations)；依据：Table 2, two-level PE Wells score；Table 1, two-level DVT Wells score。

补充交叉核验：
- GFR分层另与 [KDIGO 2024 CKD guideline](https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf) 的GFR分类核对。
- WHO贫血分层另与 [WHO 2024 原始指南](https://www.who.int/publications/i/item/9789240088542) Table 3核对。
- ADA低血糖定义核对 [ADA 2026原始表6.4](https://diabetesjournals.org/care/article-pdf/49/Supplement_1/S132/848848/dc26s006.pdf)。

边界标点修复只改变高亮span元数据；修复前后患者、指南正文、选项和闭卷输入均相同。初次筛选与错误日志保留在results/obs1_clinical/attempt1_span_boundary。