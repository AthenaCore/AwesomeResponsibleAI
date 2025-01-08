[![Awesome](awesome.svg)](https://github.com/AthenaCore/AwesomeResponsibleAI)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/AthenaCore/AwesomeResponsibleAI/graphs/commit-activity)
![GitHub](https://img.shields.io/badge/Release-PROD-yellow.svg)
![GitHub](https://img.shields.io/badge/Languages-MULTI-blue.svg)
![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)
[![GitHub](https://img.shields.io/twitter/follow/athenacoreai.svg?label=Follow)](https://twitter.com/athenacoreai)

# Awesome Responsible AI
A curated list of awesome academic research, books, code of ethics, courses, data sets, frameworks, institutes, maturity models, newsletters, principles, podcasts, reports, tools, regulations and standards related to Responsible, Trustworthy, and Human-Centered AI. 

## Main Concepts

### What is AI Governance?

AI governance is a system of rules, processes, frameworks, and tools within an organization to ensure the ethical and responsible development of AI.

### What is Human-Centered AI?

Human-Centered Artificial Intelligence (HCAI) is an approach to AI development that prioritizes human users' needs, experiences, and well-being.

### What is Open Source AI

When we refer to a “system,” we are speaking both broadly about a fully functional structure and its discrete structural elements. To be considered Open Source, the requirements are the same, whether applied to a system, a model, weights and parameters, or other structural elements.

An Open Source AI is an AI system made available under terms and in a way that grant the freedoms1 to:

- Use the system for any purpose and without having to ask for permission.
- Study how the system works and inspect its components.
- Modify the system for any purpose, including to change its output.
- Share the system for others to use with or without modifications, for any purpose.

[Source](https://opensource.org/ai/open-source-ai-definition)

### What is Responsible AI?
Responsible AI (RAI) refers to the development, deployment, and use of artificial intelligence (AI) systems in ways that are ethical, transparent, accountable, and aligned with human values. 

### What is a Responsible AI framework?
Responsible AI frameworks often encompass guidelines, principles, and practices that prioritize fairness, safety, and respect for individual rights.

### What is Trustworthy AI?

Trustworthy AI (TAI) refers to artificial intelligence systems designed and deployed to be transparent, robust and respectful of data privacy.

### Why is Responsible, Trustworthy, and Human-Centered AI important?

AI is a transformative technology prone to reshape industries, yet it requires careful governance to balance the benefits of automation and insight with protections against unintended social, economic, and security impacts. You can read more about the current wave [here](https://www.thecompendium.ai).

## Content

- [Academic Research](#academic-research)
- [Books](#books)
- [Code of Ethics](#code-of-ethics)
- [Courses](#courses)
- [Data Sets](#data-sets)
- [Frameworks](#frameworks)
- [Institutes](#institutes)
- [Maturity Models](#maturity-models)
- [Newsletters](#newsletters)
- [Principles](#principles)
- [Podcasts](#podcasts)
- [Reports](#reports)
- [Tools](#tools)
- [Regulations](#regulations)
- [Standards](#standards)
- [Citing this repository](#Citing-this-repository)

## Academic Research

### Adversarial ML

- Oprea, A., & Vassilev, A. (2023). Adversarial machine learning: A taxonomy and terminology of attacks and mitigations. National Institute of Standards and Technology. [Article](https://www.nist.gov/publications/adversarial-machine-learning-taxonomy-and-terminology-attacks-and-mitigations?utm_source=substack&utm_medium=email)

### Bias

- Schwartz, R., et al. (2022). **Towards a standard for identifying and managing bias in artificial intelligence** (Vol. 3, p. 00). US Department of Commerce, National Institute of Standards and Technology. [Article](https://www.nist.gov/publications/towards-standard-identifying-and-managing-bias-artificial-intelligence) `NIST`

### Challenges

- D'Amour, A., et al. (2022). **Underspecification presents challenges for credibility in modern machine learning**. Journal of Machine Learning Research, 23(226), 1-61. [Article](https://arxiv.org/abs/2011.03395) `Google`

### Drift

- Ackerman, S., et al. (2021, June). **Machine learning model drift detection via weak data slices**. In 2021 IEEE/ACM Third International Workshop on Deep Learning for Testing and Testing for Deep Learning (DeepTest) (pp. 1-8). IEEE. [Article](https://arxiv.org/pdf/2108.05319.pdf) `IBM`
- Ackerman, S., Raz, O., & Zalmanovici, M. (2020, February). **FreaAI: Automated extraction of data slices to test machine learning models**. In International Workshop on Engineering Dependable and Secure Machine Learning Systems (pp. 67-83). Cham: Springer International Publishing. [Article](https://arxiv.org/pdf/2108.05620.pdf) `IBM`

### Explainability

- Dhurandhar, A., Chen, P. Y., Luss, R., Tu, C. C., Ting, P., Shanmugam, K., & Das, P. (2018). **Explanations based on the missing: Towards contrastive explanations with pertinent negatives**. Advances in neural information processing systems, 31. [Article](https://papers.nips.cc/paper/7340-explanations-based-on-the-missing-towards-contrastive-explanations-with-pertinent-negatives) `University of Michigan` `IBM Research`
- Dhurandhar, A., Shanmugam, K., Luss, R., & Olsen, P. A. (2018). **Improving simple models with confidence profiles**. Advances in Neural Information Processing Systems, 31. [Article](https://papers.nips.cc/paper/8231-improving-simple-models-with-confidence-profiles) `IBM Research`
- Gurumoorthy, K. S., Dhurandhar, A., Cecchi, G., & Aggarwal, C. (2019, November). **Efficient data representation by selecting prototypes with importance weights**. In 2019 IEEE International Conference on Data Mining (ICDM) (pp. 260-269). IEEE. [Article](https://arxiv.org/abs/1707.01212) `Amazon Development Center` `IBM Research`
- Hind, M., Wei, D., Campbell, M., Codella, N. C., Dhurandhar, A., Mojsilović, A., ... & Varshney, K. R. (2019, January). **TED: Teaching AI to explain its decisions**. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society (pp. 123-129)[Article](https://doi.org/10.1145/3306618.3314273) `IBM Research`
- Lundberg, S. M., & Lee, S. I. (2017). **A unified approach to interpreting model predictions**. Advances in neural information processing systems, 30. [Article](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions), [Github](https://github.com/slundberg/shap) `University of Washington`
- Luss, R., Chen, P. Y., Dhurandhar, A., Sattigeri, P., Zhang, Y., Shanmugam, K., & Tu, C. C. (2021, August). **Leveraging latent features for local explanations**. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 1139-1149). [Article](https://arxiv.org/abs/1905.12698) `IBM Research` `University of Michigan`
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). **"Why should i trust you?" Explaining the predictions of any classifier**. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144). [Article](https://arxiv.org/abs/1602.04938), [Github](https://github.com/marcotcr/lime) `University of Washington`
- Wei, D., Dash, S., Gao, T., & Gunluk, O. (2019, May). **Generalized linear rule models**. In International conference on machine learning (pp. 6687-6696). PMLR. [Article](http://proceedings.mlr.press/v97/wei19a.html) `IBM Research`
- Contrastive Explanations Method with Monotonic Attribute Functions ([Luss et al., 2019](https://arxiv.org/abs/1905.12698))
- Boolean Decision Rules via Column Generation (Light Edition) ([Dash et al., 2018](https://papers.nips.cc/paper/7716-boolean-decision-rules-via-column-generation)) `IBM Research`
- Towards Robust Interpretability with Self-Explaining Neural Networks ([Alvarez-Melis et al., 2018](https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks)) `MIT`

### Ethical Data Products

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Iii, H. D., & Crawford, K. (2021). **Datasheets for datasets**. Communications of the ACM, 64(12), 86-92. [Article](https://arxiv.org/abs/1803.09010) `Google`
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., ... & Gebru, T. (2019, January). **Model cards for model reporting**. In Proceedings of the conference on fairness, accountability, and transparency (pp. 220-229). [Article](https://arxiv.org/abs/1810.03993) `Google`
- Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022, June). **Data cards: Purposeful and transparent dataset documentation for responsible ai**. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency (pp. 1776-1826). [Article](https://dl.acm.org/doi/10.1145/3531146.3533231) `Google`
- Rostamzadeh, N., Mincu, D., Roy, S., Smart, A., Wilcox, L., Pushkarna, M., ... & Heller, K. (2022, June). **Healthsheet: development of a transparency artifact for health datasets**. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency (pp. 1943-1961). [Article](https://arxiv.org/abs/2202.13028) `Google`
- Saint-Jacques, G., Sepehri, A., Li, N., & Perisic, I. (2020). **Fairness through Experimentation: Inequality in A/B testing as an approach to responsible design**. arXiv preprint arXiv:2002.05819. [Article](https://arxiv.org/pdf/2002.05819) `LinkedIn`

### Evaluation (of model explanations)

- Agarwal, C., et al. (2022). **Openxai: Towards a transparent evaluation of model explanations**. Advances in Neural Information Processing Systems, 35, 15784-15799. [Article](https://arxiv.org/abs/2206.11104)
- Liesenfeld, A., and Dingemanse, M. (2024). **Rethinking Open Source Generative AI: Open-Washing and the EU AI Act**. In The 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’24). Rio de Janeiro, Brazil: ACM. [Article](https://pure.mpg.de/rest/items/item_3588217_2/component/file_3588218/content) [Benchmark](https://opening-up-chatgpt.github.io)

### Fairness

- Caton, S., & Haas, C. (2024). **Fairness in machine learning: A survey.** ACM Computing Surveys, 56(7), 1-38. [Article](https://dl.acm.org/doi/full/10.1145/3616865)
- Chouldechova, A. (2017). **Fair prediction with disparate impact: A study of bias in recidivism prediction instruments**. Big data, 5(2), 153-163. [Article](https://arxiv.org/abs/1703.00056)
- Coston, A., Mishler, A., Kennedy, E. H., & Chouldechova, A. (2020, January). **Counterfactual risk assessments, evaluation, and fairness**. In Proceedings of the 2020 conference on fairness, accountability, and transparency (pp. 582-593). [Article](https://arxiv.org/abs/1909.00066) 
- Jesus, S., Saleiro, P., Jorge, B. M., Ribeiro, R. P., Gama, J., Bizarro, P., & Ghani, R. (2024). **Aequitas Flow: Streamlining Fair ML Experimentation**. arXiv preprint arXiv:2405.05809. [Article](https://arxiv.org/abs/2405.05809)
- Saleiro, P., Kuester, B., Hinkson, L., London, J., Stevens, A., Anisfeld, A., ... & Ghani, R. (2018). **Aequitas: A bias and fairness audit toolkit**. arXiv preprint arXiv:1811.05577. [Article](https://arxiv.org/abs/1811.05577)
- Vasudevan, S., & Kenthapadi, K. (2020, October). **Lift: A scalable framework for measuring fairness in ml applications**. In Proceedings of the 29th ACM international conference on information & knowledge management (pp. 2773-2780). [Article](https://arxiv.org/abs/2008.07433) `LinkedIn`

### Risk

- Slattery, P., et al. (2024). **The ai risk repository: A comprehensive meta-review, database, and taxonomy of risks from artificial intelligence**. arXiv preprint arXiv:2408.12622. [Article](https://arxiv.org/pdf/2408.12622)

### Sustainability

- Lacoste, A., Luccioni, A., Schmidt, V., & Dandres, T. (2019). **Quantifying the carbon emissions of machine learning**. arXiv preprint arXiv:1910.09700. [Article](https://arxiv.org/abs/1910.09700)
- P. Li, J. Yang, M. A. Islam, S. Ren, (2023) Making AI Less “Thirsty”: Uncovering and Addressing the Secret Water Footprint of AI Models, arXiv:2304.03271 [Article](https://arxiv.org/pdf/2304.03271)
- Parcollet, T., & Ravanelli, M. (2021). **The energy and carbon footprint of training end-to-end speech recognizers**. [Article](https://hal.archives-ouvertes.fr/hal-03190119/document)
- Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L.M., Rothchild, D., So, D., Texier, M. and Dean, J. (2021). **Carbon emissions and large neural network training**. arXiv preprint arXiv:2104.10350. [Article](https://arxiv.org/abs/2104.10350)
- Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., ... & Dennison, D. (2015). **Hidden technical debt in machine learning systems**. Advances in neural information processing systems, 28. [Article](https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) `Google`
- Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., ... & Young, M. (2014, December). **Machine learning: The high interest credit card of technical debt**. In SE4ML: software engineering for machine learning (NIPS 2014 Workshop) (Vol. 111, p. 112). [Article](https://research.google/pubs/pub43146/) `Google`
- Strubell, E., Ganesh, A., & McCallum, A. (2019). **Energy and policy considerations for deep learning in NLP**. arXiv preprint arXiv:1906.02243. [Article](https://arxiv.org/abs/1906.02243)
- Sustainable AI: AI for sustainability and the sustainability of AI ([van Wynsberghe, A. 2021](https://link.springer.com/article/10.1007/s43681-021-00043-6)). AI and Ethics, 1-6
- Green Algorithms: Quantifying the carbon emissions of computation ([Lannelongue, L. et al. 2020](https://arxiv.org/abs/2007.07610))
- C.-J. Wu, R. Raghavendra, U. Gupta, B. Acun, N. Ardalani, K. Maeng, G. Chang, F. Aga, J. Huang, C. Bai, M. Gschwind, A. Gupta, M. Ott, A. Melnikov, S. Candido, D. Brooks, G. Chauhan, B. Lee, H.-H. Lee,  K. Hazelwood, Sustainable AI: Environmental implications, challenges and opportunities in Proceedings of the 5th Conference on Machine Learning and Systems (MLSys) (2022) vol. 4, pp. 795–813. [Article](https://proceedings.mlsys.org/paper_files/paper/2022/file/462211f67c7d858f663355eff93b745e-Paper.pdf)

### Collections

- Google Research on Responsible AI: [https://research.google/pubs/?collection=responsible-ai](https://research.google/pubs/?collection=responsible-ai) `Google`
- Pipeline-Aware Fairness: [http://fairpipe.dssg.io](http://fairpipe.dssg.io)

### Reproducible/Non-Reproducible Research

**Computational reproducibility** (when the results in a paper can be replicated using the exact code and dataset provided by the authors) is becoming a significant problem not only for academic but for practitionars who want to implement AI in their organizations and aim to resuse ideas from academia. Read more about this problem [here](https://reproducible.cs.princeton.edu). 

- [Papers with Code](https://paperswithcode.com)
- [Papers without Code](https://www.paperswithoutcode.com)

## Books

### Open Access

- Barrett, M., Gerke, T. & D’Agostino McGowa, L. (2024). **Causal Inference in R** [Book](https://www.r-causal.org) `Causal Inference` `R`
- Biecek, P., & Burzykowski, T. (2021). **Explanatory model analysis: explore, explain, and examine predictive models**. Chapman and Hall/CRC. [Book](https://ema.drwhy.ai) `Explainability` `Interpretability` `Transparency` `R`
- Biecek, P. (2024). **Adversarial Model Analysis**.  [Book](https://ama.drwhy.ai) `Safety` `Red Teaming`
- Cunningham, Scott. (2021) **Causal inference: The mixtape**. Yale university press. [Book](https://mixtape.scunning.com) `Causal Inference`
- Fourrier, C. and et all. (2024) **LLM Evaluation Guidebook**. Github Repository. [Web](https://github.com/huggingface/evaluation-guidebook) `LLM Evaluation`
- Freiesleben, T. & Molnar, C. (2024). **Supervised Machine Learning for Science: How to stop worrying and love your black box.** [Book](https://ml-science-book.com/)
- Matloff, N et all. (2204) **Data Science Looks at Discrimination** [Book](https://htmlpreview.github.io/?https://github.com/matloff/dsldBook/blob/main/_book/index.html) `Fairness` `R`
- Molnar, C. (2020). **Interpretable Machine Learning**. Lulu.com. [Book](https://christophm.github.io/interpretable-ml-book/) `Explainability` `Interpretability` `Transparency` `R`
- Huntington-Klein, Nick. (2012) **The effect: An introduction to research design and causality**. Chapman and Hall/CRC. [Book](https://theeffectbook.net) `Causal Inference`

### Commercial / Propietary / Closed Access

- Trust in Machine Learning ([Varshney, K., 2022](https://www.manning.com/books/trust-in-machine-learning)) `Safety` `Privacy` `Drift` `Fairness` `Interpretability` `Explainability`
- Interpretable AI ([Thampi, A., 2022](https://www.manning.com/books/interpretable-ai)) `Explainability` `Fairness` `Interpretability` 
- AI Fairness ([Mahoney, T., Varshney, K.R., Hind, M., 2020](https://learning.oreilly.com/library/view/ai-fairness/9781492077664/) `Report` `Fairness`
- Practical Fairness ([Nielsen, A., 2021](https://learning.oreilly.com/library/view/practical-fairness/9781492075721/)) `Fairness`
- Hands-On Explainable AI (XAI) with Python ([Rothman, D., 2020](https://www.packtpub.com/product/hands-on-explainable-ai-xai-with-python/9781800208131)) `Explainability`
- AI and the Law ([Kilroy, K., 2021](https://learning.oreilly.com/library/view/ai-and-the/9781492091837/)) `Report` `Trust` `Law`
- Responsible Machine Learning ([Hall, P., Gill, N., Cox, B., 2020](https://learning.oreilly.com/library/view/responsible-machine-learning/9781492090878/)) `Report` `Law`  `Compliance` `Safety` `Privacy` 
- [Privacy-Preserving Machine Learning](https://www.manning.com/books/privacy-preserving-machine-learning)
- [Human-In-The-Loop Machine Learning: Active Learning and Annotation for Human-Centered AI](https://www.manning.com/books/human-in-the-loop-machine-learning)
- [Interpretable Machine Learning With Python: Learn to Build Interpretable High-Performance Models With Hands-On Real-World Examples](https://www.packtpub.com/product/interpretable-machine-learning-with-python/9781800203907)
- Responsible AI ([Hall, P., Chowdhury, R., 2023](https://learning.oreilly.com/library/view/responsible-ai/9781098102425/)) `Governance` `Safety` `Drift`
- Marcus, G. F. (2024). **Taming Silicon Valley: How We Can Ensure That AI Works for Us**. MIT Press. [Book](https://mitpress.mit.edu/9780262551069/taming-silicon-valley/)
- Yampolskiy, R. V. (2024) **AI: Unexplainable, Unpredictable, Uncontrollable**. 2024. CRC Press 🔗 Source [Book](https://mitpressbookstore.mit.edu/book/9781032576275)

## Code of Ethics

- [ACS Code of Professional Conduct](https://www.acs.org.au/content/dam/acs/rules-and-regulations/Code-of-Professional-Conduct_v2.1.pdf) by Australian ICT (Information and Communication Technology)
- [AI Standards Hub](https://aistandardshub.org)
- [Association for Computer Machinery's Code of Ethics and Professional Conduct](https://www.acm.org/code-of-ethics)
- [IEEE Global Initiative for Ethical Considerations in Artificial Intelligence (AI) and Autonomous Systems (AS)](https://ethicsinaction.ieee.org/)
- [ISO/IEC's Standards for Artificial Intelligence](https://www.iso.org/committee/6794475/x/catalogue/)

## Courses

### AI Alignment

- [AI Alignment](https://aisafetyfundamentals.com/alignment/) `BlueDot Impact`
- [AI Fast-Track](https://aisafetyfundamentals.com/alignment-fast-track/) `BlueDot Impact`

### AI Governance

- [AI Governance](https://aisafetyfundamentals.com/governance/) `BlueDot Impact`

### Explainability/Interpretability

- [Explainable Artificial Intelligence](https://interpretable-ml-class.github.io) `Harvard University`

### Causality

- [CS594 - Causal Inference and Learning](https://www.cs.uic.edu/~elena/courses/fall19/cs594cil.html) `University of Illinois at Chicago`

### Data/AI Ethics

- [Introduction to AI Ethics](https://www.kaggle.com/learn/intro-to-ai-ethics) `Kaggle`
- [Practical Data Ethics](https://ethics.fast.ai) `Fast.ai`

### Data Privacy

- [CS7880 - Rigorous Approaches to Data Privacy](https://www.khoury.northeastern.edu/home/jullman/cs7880s17/syllabus.html) `Northeastern University`
- [CS860 - Algorithms for Private Data Analysis](http://www.gautamkamath.com/courses/CS860-fa2022.html) `University of Waterloo`

### Ethical Design

- [CIS 4230/5230 - Ethical Algorithm Design](https://www.cis.upenn.edu/~mkearns/teaching/EADSpring24/) `University of Pennsylvania`

### Safety

- [Introduction to ML Safety](https://course.mlsafety.org) `Center for AI Safety`

## Data Sets

- [AI Risk Database](https://airisk.io/) `MITRE`
- [AI Risk Repository](https://airisk.mit.edu) `MIT`
- [ARC AGI](https://github.com/fchollet/ARC-AGI)
- [Common Corpus](https://huggingface.co/collections/PleIAs/common-corpus-65d46e3ea3980fdcd66a5613)
- [An ImageNet replacement for self-supervised pretraining without humans](https://www.robots.ox.ac.uk/~vgg/research/pass/)
- [Huggingface Data Sets](https://huggingface.co/datasets)
- [The Stack](https://www.bigcode-project.org/docs/about/the-stack/)

## Frameworks

- [A Framework for Ethical Decision Making](https://www.scu.edu/ethics/ethics-resources/a-framework-for-ethical-decision-making/) `Markkula Center for Applied Ethics`
- [Data Ethics Canvas](https://theodi.org/insights/tools/the-data-ethics-canvas-2021/) `Open Data Institute`
- [Deon](https://deon.drivendata.org) `Python` `Drivendata`
- [Ethics & Algorithms Toolkit](http://ethicstoolkit.ai)
- [RAI Toolkit](https://rai.tradewindai.com) `US Department of Defense`

## Institutes

### AI Safety Institutes (or equivalent)

- [Canada AISI](https://ised-isde.canada.ca/site/ised/en/canadian-artificial-intelligence-safety-institute) `Canada`
- [EU AI Office](https://digital-strategy.ec.europa.eu/en/policies/ai-office) `Europe`
- [Japan AISI](https://aisi.go.jp) `Japan`
- [Korea AISI](https://www.aisi.re.kr/kor) `South Korea`
- [Singapore AISI](https://www.ntu.edu.sg/dtc) `Singapore`
- [UK AISI](https://www.aisi.gov.uk) `United Kingdom`
- [US AISI](https://www.nist.gov/aisi) `United States of America`  

### Country level

- [Ada Lovelace Institute](https://www.adalovelaceinstitute.org/) `United Kingdom`   
- [Centre pour la Securité de l'IA, CeSIA](https://www.securite-ia.fr) `France`
- [European Centre for Algorithmic Transparency](https://algorithmic-transparency.ec.europa.eu/index_en)
- [Center for Human-Compatible AI](https://humancompatible.ai) `UC Berkeley` `United States of America`
- [Center for Responsible AI](https://airesponsibly.com/) `New York University` `United States of America`
- [Montreal AI Ethics Institute](https://montrealethics.ai/) `Canada`
- [Munich Center for Technology in Society (IEAI)](https://ieai.mcts.tum.de/) `TUM School of Social Sciences and Technology` `Germany`
- [National AI Centre's Responsible AI Network](https://www.industry.gov.au/science-technology-and-innovation/technology/national-artificial-intelligence-centre) `Australia` 
- [Open Data Institute](https://theodi.org/) `United Kingdom`
- [Stanford University Human-Centered Artificial Intelligence (HAI)](https://hai.stanford.edu) `United States of America`
- [The Institute for Ethical AI & Machine Learning](https://ethical.institute/)
- [UNESCO Chair in AI Ethics & Governance](https://www.ie.edu/unesco-chair-in-ai-ethics-and-governance/) `IE University` `Spain`
- [University of Oxford Institute for Ethics in AI](https://www.oxford-aiethics.ox.ac.uk/) `University of Oxford` `United Kingdom`
- [Australian Government-funded AI Adopt Centres](https://www.industry.gov.au/news/be-part-ai-revolution-ai-adopt-centres?utm_source=IGA%20Newsletter&utm_medium=Email&utm_campaign=NAIC):
  - [ARM Hub AI Adopt Centre](https://aiadopt.ai)
  - [Australian Regional AI Network (ARAIN)](https://arain.com.au)
  - [SAAM (Safe AI Adoption Model)](https://www.saam.com.au)
  - [SMEC AI (Small to Medium Enterprise Centre of Artificial Intelligence)](https://smecai.au)

## Maturity Models

### AI Governance

- [The AIGA AI Governance Framework](https://ai-governance.eu)

### Responsible AI

- [The GSMA Responsible AI Maturity Roadmap](https://www.gsma.com/solutions-and-impact/connectivity-for-good/external-affairs/responsible-ai/)

## Newsletters

- [AI Policy Perspectives](https://www.aipolicyperspectives.com)
- [AI Policy Weekly](https://aipolicyus.substack.com)
- [AI Safety Newsletter](https://newsletter.safe.ai)
- [AI Snake Oil](https://www.aisnakeoil.com)
- [Import AI](importai.substack.com)
- [Marcus on AI](https://garymarcus.substack.com)
- [Navigating AI Risks](https://www.navigatingrisks.ai)
- [One Useful Thing](https://www.oneusefulthing.org)
- [The AI Ethics Brief](https://brief.montrealethics.ai)
- [The AI Evaluation Substack](https://aievaluation.substack.com)
- [The EU AI Act Newsletter](https://artificialintelligenceact.substack.com)
- [The Machine Learning Engineer](https://ethical.institute/mle.html)
- [Turing Post](https://turingpost.substack.com)

## Principles

- [Allianz's Principles for a responsible usage of AI ](https://www.allianz.com/en/about-us/strategy-values/data-ethics-and-responsible-ai.html) `Allianz`
- [Asilomar AI principles](https://futureoflife.org/open-letter/ai-principles/)
- [European Commission's Guidelines for Trustworthy AI](https://ec.europa.eu/futurium/en/ai-alliance-consultation)
- [Google's AI Principles](https://ai.google/principles/) `Google`
- [IEEE's Ethically Aligned Design](https://ethicsinaction.ieee.org/) `IEEE`
- [Microsoft's AI principles](https://www.microsoft.com/en-us/ai/responsible-ai) `Microsoft`
- [OECD's AI principles](https://oecd.ai/en/ai-principles) `OECD`
- [Telefonica's AI principles](https://www.telefonica.com/en/sustainability-innovation/how-we-work/business-principles/#artificial-intelligence-principles) `Telefonica`
- [The Institute for Ethical AI & Machine Learning: The Responsible Machine Learning Principles](https://ethical.institute/principles.html)

Additional:

- [FAIR Principles](https://www.go-fair.org/fair-principles/) `Findability` `Accessibility` `Interoperability` `Reuse`

## Podcasts

- [The Human-Centered AI Podcast](https://podcasts.apple.com/us/podcast/the-human-centered-ai-podcast/id1499839858)
- [Responsible AI Podcast](https://open.spotify.com/show/63Fx70r96P3ghWavisvPEQ)
- [Trustworthy AI](https://marketing.truera.com/trustworthy-ai-podcast)

## Reports

### AI Governance

- Araujo, R. 2024. **Understanding the First Wave of AI Safety Institutes: Characteristics, Functions, and Challenges**. Institute for AI Policy and Strategy (IAPS) [Article](https://www.iaps.ai/research/understanding-aisis)
- Buchanan, B. 2020. **The AI triad and what it means for national security strategy**. Center for Security and Emerging Technology. [Article](https://cset.georgetown.edu/wp-content/uploads/CSET-AI-Triad-Report.pdf)
- Corrigan, J. et al. 2023. **The Policy Playbook: Building a Systems-Oriented Approach to Technology and National Security Policy**. CSET (Center for Security and Emerging Technology) [Article](https://cset.georgetown.edu/publication/the-policy-playbook/)
- Curto, J. 2024. **How Can Spain Remain Internationally Competitive in AI under EU Legislation?** [Article](https://github.com/AthenaCore/AwesomeResponsibleAI/blob/main/How%20Can%20Spain%20Remain%20Internationally%20Competitive%20in%20AI%20under%20EU%20Legislation.pdf)
- CSIS. 2024 **The AI Safety Institute International Network: Next Steps and Recommendations**. CSIS (Center for Strategic and International Studies) [Article](https://www.csis.org/analysis/ai-safety-institute-international-network-next-steps-and-recommendations)
- Gupta, Ritwik, et al. (2024). **Data-Centric AI Governance: Addressing the Limitations of Model-Focused Policies**. arXiv preprint arXiv:2409.17216 (Article)[https://arxiv.org/pdf/2409.17216]
- Hendrycks, D. et al. 2023. **An overview of catastrophic AI risks. Center of AI Safety**.  arXiv preprint arXiv:2306.12001. [Article](https://arxiv.org/pdf/2306.12001)
- Janjeva, A., et al. (2023). **Strengthening Resilience to AI Risk. A guide for UK policymakers**. CETaS (Centre for Emerging Technology and Security) [Article](https://ceas.turing.ac.uk/sites/default/files/2023-08/cetas-cltr_ai_risk_briefing_paper.pdf)
- Piattini, M. and Fernández C.M. 2024. **Marco Confiable**. Revista SIC 162 [Article](https://revistasic.es/revista-sic/sic-162/colaboraciones/marco-confiable/)
- Sastry, G., et al. 2024. **Computing Power and the Governance of Artificial Intelligence**. arXiv preprint arXiv:2402.08797. [Article]( https://arxiv.org/pdf/2402.08797)

### (AI) Incidents databases

- [AI Incident Database](https://incidentdatabase.ai)
- [AI Vulnerability Database (AVID)](https://avidml.org/)
- [AIAAIC](https://www.aiaaic.org/)
- [AI Badness: An open catalog of generative AI badness](https://badness.ai/)
- [George Washington University Law School's AI Litigation Database](https://blogs.gwu.edu/law-eti/ai-litigation-database/)
- [Merging AI Incidents Research with Political Misinformation Research: Introducing the Political Deepfakes Incidents Database](https://osf.io/fvqg3/)
- [OECD AI Incidents Monitor](https://oecd.ai/en/incidents)
- [Verica Open Incident Database (VOID)](https://www.thevoid.community/)

### Market Analysis

- [State of AI](https://www.stateof.ai) - from 2018 up to now -
- [The AI Index Report](https://aiindex.stanford.edu) - from 2017 up to now - `Stanford Institute for Human-Centered Artificial Intelligence`
- [AI Safety Index](https://futureoflife.org/document/fli-ai-safety-index-2024/) - 2024 - `Future of Life`

### Other

- [Four Principles of Explainable Artificial Intelligence](https://nvlpubs.nist.gov/nistpubs/ir/2021/NIST.IR.8312.pdf) `NIST` `Explainability`
- [Psychological Foundations of Explainability and Interpretability in Artificial Intelligence](https://nvlpubs.nist.gov/nistpubs/ir/2021/NIST.IR.8367.pdf) `NIST` `Explainability`
- [Inferring Concept Drift Without Labeled Data, 2021](https://concept-drift.fastforwardlabs.com) `Drift`
- [Interpretability, Fast Forward Labs, 2020](https://ff06-2020.fastforwardlabs.com) `Interpretability`
- [Towards a Standard for Identifying and Managing Bias in Artificial Intelligence (NIST Special Publication 1270)](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1270.pdf) `NIST` `Bias`
- [Auditing machine learning algorithms](https://www.auditingalgorithms.net/index.html) `Auditing`

## Tools

### Assessments

- [The Assessment List for Trustworthy Artificial Intelligence](https://altai.insight-centre.org)

### Bias

- [balance](https://import-balance.org) `Python` `Facebook`
- [smclafify](https://github.com/aws/amazon-sagemaker-clarify) `Python` `Amazon`
- [SolasAI](https://github.com/SolasAI/solas-ai-disparity) `Python`
- [TRAK (Attributing Model Behaviour at Scale)](https://github.com/MadryLab/trak) [Article](https://arxiv.org/pdf/2303.14186) `Python`

### Causal Inference

- [CausalAI](https://github.com/salesforce/causalai) `Python` `Salesforce`
- [CausalNex](https://causalnex.readthedocs.io) `Python`
- [CausalImpact](https://cran.r-project.org/web/packages/CausalImpact) `R`
- [Causalinference](https://causalinferenceinpython.org) `Python`
- [Causal Inference 360](https://github.com/BiomedSciAI/causallib) `Python`
- [CausalPy](https://github.com/pymc-labs/CausalPy) `Python`
- [CIMTx: Causal Inference for Multiple Treatments with a Binary Outcome](https://cran.r-project.org/web/packages/CIMTx) `R`
- [dagitty](https://cran.r-project.org/web/packages/dagitty) `R`
- [DoWhy](https://github.com/Microsoft/dowhy) `Python` `Microsoft`
- [mediation: Causal Mediation Analysis](https://cran.r-project.org/web/packages/mediation) `R`
- [MRPC](https://cran.r-project.org/web/packages/MRPC) `R`

### Drift

- [Alibi Detect](https://github.com/SeldonIO/alibi-detect) `Python`
- [Deepchecks](https://github.com/deepchecks/deepchecks) `Python`
- [drifter](https://cran.r-project.org/web/packages/drifter/) `R`
- [Evidently](https://github.com/evidentlyai/evidently) `Python`
- [nannyML](https://github.com/NannyML/nannyml) `Python`
- [phoenix](https://github.com/Arize-ai/phoenix) `Python`

### Fairness

- [Aequitas' Bias & Fairness Audit Toolkit](http://aequitas.dssg.io/) `Python`
- [AI360 Toolkit](https://github.com/Trusted-AI/AIF360) `Python` `R` `IBM`
- [dsld: Data Science Looks at Discrimination](https://cran.r-project.org/web/packages/dsld/index.html) `R`
- [EDFfair: Explicitly Deweighted Features](https://github.com/matloff/EDFfair) `R`
- [EquiPy](https://github.com/equilibration/equipy) `Python`
- [Fairlearn](https://fairlearn.org) `Python` `Microsoft`
- [Fairmodels](https://fairmodels.drwhy.ai) `R` `University of California`
- [fairness](https://cran.r-project.org/web/packages/fairness/) `R`
- [Fairness Indicators](https://github.com/tensorflow/fairness-indicators) `Python` `Google`
- [FairRankTune](https://kcachel.github.io/fairranktune/) `Python`
- [FairPAN - Fair Predictive Adversarial Network](https://modeloriented.github.io/FairPAN/) `R`
- [OxonFair](https://github.com/oxfordinternetinstitute/oxonfair) `Python` `Oxford Internet Institute`
- [Themis ML](https://github.com/cosmicBboy/themis-ml) `Python`
- [What-If Tool](https://github.com/PAIR-code/what-if-tool) `Python` `Google`

### Interpretability/Explicability

- [Alibi Explain](https://github.com/SeldonIO/alibi) `Python`
- [Automated interpretability](https://github.com/openai/automated-interpretability) `Python` `OpenAI`
- [AI360 Toolkit](https://github.com/Trusted-AI/AIF360) `Python` `R` `IBM`
- [aorsf: Accelerated Oblique Random Survival Forests](https://cran.r-project.org/web/packages/aorsf/index.html) `R`
- [breakDown: Model Agnostic Explainers for Individual Predictions](https://cran.r-project.org/web/packages/breakDown/index.html) `R`
- [captum](https://github.com/pytorch/captum) `Python` `PyTorch`
- [ceterisParibus: Ceteris Paribus Profiles](https://cran.r-project.org/web/packages/ceterisParibus/index.html) `R`
- [DALEX: moDel Agnostic Language for Exploration and eXplanation](https://dalex.drwhy.ai) `Python` `R`
- [DALEXtra: extension for DALEX](https://modeloriented.github.io/DALEXtra) `Python` `R`
- [Dianna](https://github.com/dianna-ai/dianna) `Python`
- [Diverse Counterfactual Explanations (DiCE)](https://github.com/interpretml/DiCE) `Python` `Microsoft`
- [dtreeviz](https://github.com/parrt/dtreeviz) `Python` 
- [ecco](https://pypi.org/project/ecco/) [article](https://jalammar.github.io/explaining-transformers/) `Python`
- [effectplots](https://github.com/mayer79/effectplots) `R`
- [eli5](https://github.com/TeamHG-Memex/eli5) `Python`
- [explabox](https://explabox.readthedocs.io/en/latest/index.html) `Python` `National Police Lab AI`
- [eXplainability Toolbox](https://ethical.institute/xai.html) `Python`
- [ExplainaBoard](https://github.com/neulab/ExplainaBoard) `Python` `Carnegie Mellon University`
- [ExplainerHub](https://explainerdashboard.readthedocs.io/en/latest/index.html) [in github](https://github.com/oegedijk/explainerdashboard) `Python` 
- [fastshap](https://github.com/bgreenwell/fastshap) `R`
- [fasttreeshap](https://github.com/linkedin/fasttreeshap) `Python` `LinkedIn`
- [FAT Forensics](https://fat-forensics.org/) `Python`
- [ferret](https://github.com/g8a9/ferret) `Python`
- [flashlight](https://github.com/mayer79/flashlight) `R`
- [Human Learn](https://github.com/koaning/human-learn) `Python`
- [hstats](https://cran.r-project.org/web/packages/hstats/index.html) `R`
- [innvestigate](https://github.com/albermax/innvestigate) `Python` `Neural Networks`
- [Inseq](https://github.com/inseq-team/inseq) `Python`
- [intepretML](https://interpret.ml) `Python`
- [interactions: Comprehensive, User-Friendly Toolkit for Probing Interactions](https://cran.r-project.org/web/packages/interactions/index.html) `R`
- [kernelshap: Kernel SHAP](https://cran.r-project.org/web/packages/kernelshap/index.html) `R`
- [Learning Interpretability Tool](https://pair-code.github.io/lit/) `Python` `Google`
- [lime: Local Interpretable Model-Agnostic Explanations](https://cran.r-project.org/web/packages/lime/index.html) `R`
- [Network Dissection](http://netdissect.csail.mit.edu) `Python` `Neural Networks` `MIT` 
- [OmniXAI](https://github.com/salesforce/OmniXAI) `Python` `Salesforce`
- [Shap](https://github.com/slundberg/shap) `Python`
- [Shapash](https://github.com/maif/shapash) `Python`
- [shapper](https://cran.r-project.org/web/packages/shapper/index.html) `R`
- [shapviz](https://cran.r-project.org/web/packages/shapviz/index.html) `R`
- [Skater](https://github.com/oracle/Skater) `Python` `Oracle`
- [survex](https://github.com/ModelOriented/survex) `R`
- [teller](https://github.com/Techtonique/teller) `Python`
- [TCAV (Testing with Concept Activation Vectors)](https://pypi.org/project/tcav/) `Python` 
- [Transformer Debugger](https://github.com/openai/transformer-debugger) `Python` `OpenAI` 
- [truelens](https://pypi.org/project/trulens/) `Python` `Truera`
- [truelens-eval](https://pypi.org/project/trulens-eval/) `Python` `Truera`
- [pre: Prediction Rule Ensembles](https://cran.r-project.org/web/packages/pre/index.html) `R`
- [Vetiver](https://rstudio.github.io/vetiver-r/) `R` `Python` `Posit`
- [vip](https://github.com/koalaverse/vip) `R`
- [vivid](https://cloud.r-project.org/web/packages/vivid/index.html) `R`
- [XAI - An eXplainability toolbox for machine learning](https://github.com/EthicalML/xai) `Python` `The Institute for Ethical Machine Learning`
- [xplique](https://github.com/deel-ai/xplique) `Python`
- [XAIoGraphs](https://github.com/Telefonica/XAIoGraphs) `Python` `Telefonica`
- [Zennit](https://github.com/chr5tphr/zennit) `Python`

### Interpretable Models

- [imodels](https://github.com/csinva/imodels) `Python`
- [imodelsX](https://github.com/csinva/imodelsX) `Python`
- [interpretML](https://github.com/interpretml/interpret) `Python` `Microsoft` [`R`](https://cran.r-project.org/web/packages/interpret/index.html)
- [PiML Toolbox](https://github.com/SelfExplainML/PiML-Toolbox) `Python`
- [Tensorflow Lattice](https://github.com/tensorflow/lattice) `Python` `Google`

### LLM Regulation Compliance

- [COMPL-AI](https://compl-ai.org) `Python` `ETH Zurich` `Insait` `LaticeFlow AI`

### LLM Evaluations and Benchmarks

- [AIluminate](https://mlcommons.org/ailuminate/)
- [AlignEval: Making Evals Easy, Fun, and Semi-Automated](https://aligneval.com) [Motivation](https://eugeneyan.com/writing/aligneval/)
- [Azure AI Evaluation](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/evaluation/azure-ai-evaluation) `Python` `Microsoft`
- [BALROG](https://github.com/balrog-ai/BALROG) `Python` 
- [DeepEval](https://github.com/confident-ai/deepeval) `Python` 
- [evals](https://github.com/openai/evals) `Python` `OpenAI`
- [FBI: Finding Blindspots in LLM Evaluations with Interpretable Checklists](https://github.com/AI4Bharat/FBI) `Python`
- [FrontierMath](https://epoch.ai/frontiermath)
- [Geekbench AI](https://www.geekbench.com/ai/)
- [Giskard](https://github.com/Giskard-AI/giskard) `Python`
- [Inspect](https://ukgovernmentbeis.github.io/inspect_ai/) `UK AISI` `Python`
- [Jailbreakbench](https://jailbreakbench.github.io) `Python`
- [LightEval](https://github.com/huggingface/lighteval) `HuggingFace` `Python`
- [LiveBench: A Challenging, Contamination-Free LLM Benchmark](https://livebench.ai) `Contamination free`
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) `Python`
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) `Python`
- [MixEval](https://mixeval.github.io) `Python`
- [ML Commons Safety Benchmark for general purpose AI chat model](https://mlcommons.org/benchmarks/ai-safety/general_purpose_ai_chat_benchmark/)
- [MLPerf Training Benchmark](https://mlcommons.org/benchmarks/training/) `Training`
- [MMMU](https://github.com/MMMU-Benchmark/MMMU) `Apple` `Python`
- [Moonshoot](https://github.com/aiverify-foundation/moonshot) `AI Verify Foundation` `Python`
- [NaturalBench](https://linzhiqiu.github.io/papers/naturalbench/) `Python`
- [opik](https://github.com/comet-ml/opik) `Comet` `Python`
- [Phoenix](https://github.com/Arize-ai/phoenix) `Arize AI` `Python`
- [Prometheus](https://github.com/prometheus-eval/prometheus) `Python`
- [Promptfoo](https://github.com/promptfoo/promptfoo) `Python`
- [ragas](https://github.com/explodinggradients/ragas) `Python`
- [Rouge](https://pypi.org/project/rouge/) `Python`
- [SALAD-BENCH](https://github.com/OpenSafetyLab/SALAD-BENCH) [Article](https://arxiv.org/abs/2402.05044) `Python`
- [simple evals](https://github.com/openai/simple-evals) `Python` `OpenAI`
- [StrongREJECT jailbreak benchmark](https://github.com/dsbowen/strong_reject) `Python`
- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://github.com/sierra-research/tau-bench) `Python`
- [Yet Another Applied LLM Benchmark](https://github.com/carlini/yet-another-applied-llm-benchmark) `Python`
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) `Python`
- [WindowsAgentArena](https://github.com/microsoft/windowsagentarena) `Python` `Microsoft`

Additional benchmarks can be found [here](https://airtable.com/app83SBBFk9WO25hJ/shrSs3bXSx2bBDrso/tblNpnE4pzBnaC5lT?viewControls=on).

### Performance (& Automated ML)

- [auditor](https://github.com/ModelOriented/auditor) `R`
- [automl: Deep Learning with Metaheuristic](https://cran.r-project.org/web/packages/automl/index.html) `R`
- [AutoKeras](https://github.com/keras-team/autokeras) `Python`
- [Auto-Sklearn](https://github.com/automl/auto-sklearn) `Python`
- [DataPerf](https://sites.google.com/mlcommons.org/dataperf/) `Python` `Google`
- [deepchecks](https://deepchecks.com) `Python`
- [EloML](https://github.com/ModelOriented/EloML) `R`
- [Featuretools](https://www.featuretools.com) `Python`
- [LOFO Importance](https://github.com/aerdem4/lofo-importance) `Python`
- [forester](https://modeloriented.github.io/forester/) `R`
- [metrica: Prediction performance metrics](https://adriancorrendo.github.io/metrica/) `R`
- [model-diagnostics](https://github.com/lorentzenchr/model-diagnostics) `Python`
- [NNI: Neural Network Intelligence](https://github.com/microsoft/nni) `Python` `Microsoft`
- [performance](https://github.com/easystats/performance) `R`
- [rliable](https://github.com/google-research/rliable) `Python` `Google`
- [SLmetrics](https://github.com/serkor1/SLmetrics/) `R`
- [TensorFlow Model Analysis](https://github.com/tensorflow/model-analysis) `Python` `Google`
- [TPOT](http://epistasislab.github.io/tpot/) `Python`
- [Unleash](https://www.getunleash.io) `Python`
- [Yellowbrick](https://www.scikit-yb.org/en/latest/) `Python`
- [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher) ([Examples](https://github.com/CalculatedContent/WeightWatcher-Examples)) `Python`

### (AI/Data) Poisoning

- [Copyright Traps for Large Language Models](https://github.com/computationalprivacy/copyright-traps) `Python`
- [Nightshade](https://nightshade.cs.uchicago.edu) `University of Chicago` `Tool`
- [Glaze](https://glaze.cs.uchicago.edu) `University of Chicago` `Tool`
- [Fawkes](http://sandlab.cs.uchicago.edu/fawkes/) `University of Chicago` `Tool`

### Privacy

- [BackPACK](https://toiaydcdyywlhzvlob.github.io/backpack) `Python`
- [DataSynthesizer: Privacy-Preserving Synthetic Datasets](https://github.com/DataResponsibly/DataSynthesizer) `Python` `Drexel University` `University of Washington`
- [diffpriv](https://github.com/brubinstein/diffpriv) `R`
- [Diffprivlib](https://github.com/IBM/differential-privacy-library) `Python` `IBM`
- [Discrete Gaussian for Differential Privacy](https://github.com/IBM/discrete-gaussian-differential-privacy) `Python` `IBM`
- [Opacus](https://opacus.ai) `Python` `Facebook`
- [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) `Python` `National University of Singapore`
- [PyVacy: Privacy Algorithms for PyTorch](https://github.com/ChrisWaites/pyvacy) `Python`
- [SEAL](https://github.com/Microsoft/SEAL) `Python` `Microsoft`
- [SmartNoise](https://github.com/opendp/smartnoise-core) `Python` `OpenDP`
- [Tensorflow Privacy](https://github.com/tensorflow/privacy) `Python` `Google`

### Reliability Evaluation (of post hoc explanation methods and LLMs evaluations) 

- [BELLS (Benchmark for the Evaluation of LLM Safeguards)](https://github.com/CentreSecuriteIA/BELLS) `Python` `CeSIA - Centre pour la Sécurité de l'IA`
- [BetterBench](https://betterbench.stanford.edu) [Database](https://betterbench.stanford.edu/database.html)
- [openXAI](https://open-xai.github.io) `Python`

### Robustness

- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) `Python`
- [BackdoorBench](https://github.com/SCLBD/BackdoorBench) `Python`
- [Factool](https://github.com/GAIR-NLP/factool) `Python`
- [Foolbox](https://github.com/bethgelab/foolbox) `Python`
- [Guardrails](https://github.com/guardrails-ai/guardrails) `Python`
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) `Python` `Amazon`

### Safety

- [AIxploit](https://github.com/AINTRUST-AI/aixploit) `Python`
- [Diotra](https://github.com/usnistgov/dioptra) `Python` `NIST`
- [Garak](https://github.com/NVIDIA/garak) `Python` `Nvidia`

### Security

- [Counterfit](https://github.com/Azure/counterfit/) `Python` `Microsoft`
- [Modelscan](https://github.com/protectai/modelscan) `Python`
- [NB Defense](https://nbdefense.ai) `Python`
- [PyRIT](https://github.com/Azure/PyRIT) `Python` `Microsoft`
- [Rebuff Playground](https://www.rebuff.ai/playground) `Python`
- [Turing Data Safe Haven](https://github.com/alan-turing-institute/data-safe-haven) `Python` `The Alan Turing Institute`

For consumers:

- [Data Breach](https://databreach.com)
- [Have I been pwned?](https://haveibeenpwned.com)
- [Have I Been Trained?](https://haveibeentrained.com)

### Sustainability

- [Azure Sustainability Calculator](https://appsource.microsoft.com/en-us/product/power-bi/coi-sustainability.sustainability_dashboard) `Microsoft`
- [Carbon Tracker](https://github.com/lfwa/carbontracker) [Website](https://carbontracker.info) `Python`
- [CodeCarbon](https://github.com/mlco2/codecarbon) [Website](https://codecarbon.io) `Python`
- [Computer Progress](https://www.computerprogress.com)
- [Impact Framework](https://if.greensoftware.foundation) `API`

### (RAI) Toolkit

- [Deepchecks](https://github.com/deepchecks/deepchecks) `Python`
- [Dr. Why](https://github.com/ModelOriented/DrWhy) `R` `Warsaw University of Technology`
- [Mercury](https://www.bbvaaifactory.com/mercury/) `Python` `BBVA`
- [Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox) `Python` `Microsoft`
- [Responsible AI Widgets](https://github.com/microsoft/responsible-ai-widgets) `R` `Microsoft`
- [The Data Cards Playbook](https://pair-code.github.io/datacardsplaybook/) `Python` `Google`
- [Zeno Hub](https://github.com/zeno-ml/zeno-hub) `Python`

### (AI) Watermarking

- [AudioSeal: Proactive Localized Watermarking](https://github.com/facebookresearch/audioseal) `Python` `Facebook`
- [MarkLLM: An Open-Source Toolkit for LLM Watermarking](https://github.com/thu-bpm/markllm) `Python`
- [SynthID Text](https://github.com/google-deepmind/synthid-text) `Python` `Google`

## Regulations

### Definition

**What are regulations?**

Regulations are requirements established by governments.

### Interesting resources

- [Data Protection and Privacy Legislation Worldwide](https://unctad.org/page/data-protection-and-privacy-legislation-worldwide) `UNCTAD`
- [Data Protection Laws of the Word](https://www.dlapiperdataprotection.com)
- [Digital Policy Alert](https://digitalpolicyalert.org/analysis)
- [ETO Agora](emerging-technology-observatory/)
- [GDPR Comparison](https://www.activemind.legal/law/)
- [National AI policies & strategies](https://oecd.ai/en/dashboards/overview)
- [Policy Database](https://aistandardshub.org/policy-and-strategy-search/)
- [SCL Artificial Intelligence Contractual Clauses](https://www.scl.org/wp-content/uploads/2024/02/AI-Clauses-Project-October-2023-final-1.pdf)

### Australia

- [AI and ESG](https://www.industry.gov.au/publications/ai-and-esg)
- [The AI Impact Navigator](https://www.industry.gov.au/publications/ai-impact-navigator)
- [Voluntary AI Safety Standard](https://www.industry.gov.au/publications/voluntary-ai-safety-standard)

### Canada

- [Algorithmic Impact Assessment tool](https://www.canada.ca/en/government/system/digital-government/digital-government-innovations/responsible-use-ai/algorithmic-impact-assessment.html)
- [Directive on Automated Decision-Making](https://www.tbs-sct.gc.ca/pol/doc-eng.aspx?id=32592)
- [Directive on Privacy Practices](https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=18309)
- [Directive on Security Management](https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32611)
- [Directive on Service and Digital](https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32601)
- [Policy on Government Security](https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=16578)
- [Policy on Service and Digital](https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32603)
- [Privacy Act](https://laws-lois.justice.gc.ca/eng/ACTS/P-21/)

### European Union

Short Name | Code | Description | Status | Website | Legal text
---|---|---|---|---|---
Cyber Resilience Act (CRA) - horizontal cybersecurity requirements for products with digital elements | 2022/0272(COD)| It introduces mandatory cybersecurity requirements for hardware and software products, throughout their whole lifecycle. | Proposal | [Website](https://digital-strategy.ec.europa.eu/en/policies/cyber-resilience-act) |[Source](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ%3AL_202402847)
Data Act | EU/2023/2854 | It enables a fair distribution of the value of data by establishing clear and fair rules for accessing and using data within the European data economy. | Published | [Website](https://digital-strategy.ec.europa.eu/en/policies/data-act)| [Source](https://eur-lex.europa.eu/eli/reg/2023/2854)
Data Governance Act | EU/2022/868 | It supports the setup and development of Common European Data Spaces in strategic domains, involving both private and public players, in sectors such as health, environment, energy, agriculture, mobility, finance, manufacturing, public administration and skills. | Published | [Website](https://digital-strategy.ec.europa.eu/en/policies/data-governance-act) | [Source](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R0868)
Digital Market Act | EU/2022/1925 | It establishes a set of clearly defined objective criteria to identify “gatekeepers”. Gatekeepers are large digital platforms providing so called core platform services, such as for example online search engines, app stores, messenger services. Gatekeepers will have to comply with the do’s (i.e. obligations) and don’ts (i.e. prohibitions) listed in the DMA. | Published | [Website](https://digital-markets-act.ec.europa.eu/index_en) | [Source](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32022R1925)
Digital Services Act | EU/2022/2026 | It regulates online intermediaries and platforms such as marketplaces, social networks, content-sharing platforms, app stores, and online travel and accommodation platforms. Its main goal is to prevent illegal and harmful activities online and the spread of disinformation. It ensures user safety, protects fundamental rights, and creates a fair and open online platform environment. | Published | [Website](https://commission.europa.eu/strategy-and-policy/priorities-2019-2024/europe-fit-digital-age/digital-services-act_en) | [Source](https://eur-lex.europa.eu/legal-content/EN/TXT/?toc=OJ%3AL%3A2022%3A277%3ATOC&uri=uriserv%3AOJ.L_.2022.277.01.0001.01.ENG)
DMS Directive | EU/2019/790 | It is intended to ensure a well-functioning marketplace for copyright. | Published | [Website](https://digital-strategy.ec.europa.eu/en/policies/copyright-legislation) | [Source](https://eur-lex.europa.eu/eli/dir/2019/790/oj)
Energy Efficiency Directive | EU/2023/1791 | It establishes ‘energy efficiency first’ as a fundamental principle of EU energy policy, giving it legal-standing for the first time. In practical terms, this means that energy efficiency must be considered by EU countries in all relevant policy and major investment decisions taken in the energy and non-energy sectors. | Published | [Website](https://energy.ec.europa.eu/topics/energy-efficiency/energy-efficiency-targets-directive-and-rules/energy-efficiency-directive_en) | [Source](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ%3AJOL_2023_231_R_0001&qid=1695186598766)
EU AI ACT | EU/2024/1689 | It assigns applications of AI to three risk categories. First, applications and systems that create an unacceptable risk are banned. Second, high-risk applications are subject to specific legal requirements. Lastly, applications not explicitly banned or listed as high-risk are largely left unregulated. | Published | [Website](https://artificialintelligenceact.eu) | [Source](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401689)
General Data Protection Regulation (GDPR) | EU/2016/679 | It strengthens individuals' fundamental rights in the digital age and facilitate business by clarifying rules for companies and public bodies in the digital single market. | Published | [Website](https://gdpr.eu/) | [Source](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32016R0679)
NIS2 Directive | EU/2022/2555 |It provides legal measures to boost the overall level of cybersecurity in the EU by ensuring preparedness, cooperation and security cultere across the Member States. | Published | [Website](https://digital-strategy.ec.europa.eu/en/policies/nis2-directive) | [Source](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32022L2555)

- [Hiroshima Process International Guiding Principles for Advanced AI system](https://digital-strategy.ec.europa.eu/en/library/hiroshima-process-international-guiding-principles-advanced-ai-system)

### Singapore

- [Singapore’s Approach to AI Governance - Verify](https://www.pdpc.gov.sg/help-and-resources/2020/01/model-ai-governance-framework)

### United States

- State consumer privacy laws: California ([CCPA](https://www.oag.ca.gov/privacy/ccpa) and its amendment, [CPRA](https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202120220AB1490)), Virginia ([VCDPA](https://lis.virginia.gov/cgi-bin/legp604.exe?212+sum+HB2307)), and Colorado ([ColoPA](https://leg.colorado.gov/sites/default/files/documents/2021A/bills/2021a_190_rer.pdf)).
- Specific and limited privacy data laws: [HIPAA](https://www.cdc.gov/phlp/publications/topic/hipaa.html), [FCRA](https://www.ftc.gov/enforcement/statutes/fair-credit-reporting-act), [FERPA](https://www.cdc.gov/phlp/publications/topic/ferpa.html), [GLBA](https://www.ftc.gov/tips-advice/business-center/privacy-and-security/gramm-leach-bliley-act), [ECPA](https://bja.ojp.gov/program/it/privacy-civil-liberties/authorities/statutes/1285), [COPPA](https://www.ftc.gov/enforcement/rules/rulemaking-regulatory-reform-proceedings/childrens-online-privacy-protection-rule), [VPPA](https://www.law.cornell.edu/uscode/text/18/2710) and [FTC](https://www.ftc.gov/enforcement/statutes/federal-trade-commission-act).
- [EU-U.S. and Swiss-U.S. Privacy Shield Frameworks](https://www.privacyshield.gov/welcome) - The EU-U.S. and Swiss-U.S. Privacy Shield Frameworks were designed by the U.S. Department of Commerce and the European Commission and Swiss Administration to provide companies on both sides of the Atlantic with a mechanism to comply with data protection requirements when transferring personal data from the European Union and Switzerland to the United States in support of transatlantic commerce.
- [Executive Order on Maintaining American Leadership in AI](https://www.whitehouse.gov/presidential-actions/executive-order-maintaining-american-leadership-artificial-intelligence/) - Official mandate by the President of the US to 
[Privacy Act of 1974](https://www.justice.gov/opcl/privacy-act-1974) - The privacy act of 1974 which establishes a code of fair information practices that governs the collection, maintenance, use and dissemination of information about individuals that is maintained in systems of records by federal agencies.
- [Privacy Protection Act of 1980](https://epic.org/privacy/ppa/) - The Privacy Protection Act of 1980 protects journalists from being required to turn over to law enforcement any work product and documentary materials, including sources, before it is disseminated to the public.
- [AI Bill of Rights](https://www.whitehouse.gov/ostp/ai-bill-of-rights/) - The Blueprint for an AI Bill of Rights is a guide for a society that protects all people from IA threats based on five principles: Safe and Effective Systems, Algorithmic Discrimination Protections, Data Privacy, Notice and Explanation, and  Human Alternatives, Consideration, and Fallback.

## Standards

### Definition

**What are standards?**

Standards are **voluntary**, **consensus solutions**. They document an **agreement** on how a material, product, process, or service should be **specified**, **performed** or **delivered**. They keep people safe and **ensure things work**. They create **confidence** and provide **security** for investment.

Standards can be understood as formal specifications of best practices as well. There is a growing number of standards related to AI. You can search for the latest in the [Standards Database](https://aistandardshub.org/ai-standards-search/) from [AI Standards Hub](https://aistandardshub.org).

###  Standards

### CEN Standards

The European Committee for Standardization is one of three European Standardization Organizations (together with CENELEC and ETSI) that have been officially recognized by the European Union and by the European Free Trade Association (EFTA) as being responsible for developing and defining voluntary standards at European level.

Domain | Standard | Status | URL
---|---|---|---
Data governance and quality for AI within the European context | CEN/CLC/TR 18115:2024 | Published | [Source]

CEN AI Work programme can be found [here](https://standards.cencenelec.eu/dyn/www/f?p=205:22:0::::FSP_ORG_ID,FSP_LANG_ID:2916257,25&cs=1827B89DA69577BF3631EE2B6070F207D).

### IEEE Standards

Domain | Standard | Status | URL
---|---|---|---
IEEE Guide for an Architectural Framework for Explainable Artificial Intelligence | IEEE 2894-2024 | Published | [Source](https://standards.ieee.org/ieee/2894/11296/)
IEEE Standard for Ethical Considerations in Emulated Empathy in Autonomous and Intelligent Systems | IEEE 7014-2024 | Published | [Source](https://www.iso.org/standard/74296.html)

### UNE Standards

[UNE](https://www.en.une.org) is Spain's only Standardisation Organisation, designated by the Spanish Ministry of Economy, Industry and Competitiveness to the European Commission. It helps Spanish organizations to keep up-to-date on all aspects related to standardisation:​​​​

- Discover the new regulatory developments;
- Take part in developing standards;
- Learn how to integrate standardisation in your R&D&i project;

Domain | Standard | Status | URL
---|---|---|---
Calidad del dato | UNE 0079:2023 | Published | [Source](https://tienda.aenor.com/norma-une-especificacion-une-0079-2023-n0071118)
Gestión del dato | UNE 0078:2023 | Published | [Source](https://tienda.aenor.com/norma-une-especificacion-une-0078-2023-n0071117)
Gobierno del dato | UNE 0077:2023 | Published | [Source](https://tienda.aenor.com/norma-une-especificacion-une-0077-2023-n0071116)
Guía de evaluación de la Calidad de un Conjunto de Datos. | UNE 0081:2023 | Published | [Source](https://tienda.aenor.com/norma-une-especificacion-une-0081-2023-n0071807)
Guía de evaluación del Gobierno, Gestión y Gestión de la Calidad del Dato. | UNE 0080:2023 | Published | [Source](https://tienda.aenor.com/norma-une-especificacion-une-0080-2023-n0071383)

Additional translations in Spanish can be found [here](https://tienda.aenor.com/Buscador).

### ISO/IEC Standards

Domain | Standard | Status | URL
---|---|---|---
AI Concepts and Terminology| ISO/IEC 22989:2022 Information technology — Artificial intelligence — Artificial intelligence concepts and terminology | Published | https://www.iso.org/standard/74296.html
AI Risk Management | ISO/IEC 23894:2023 Information technology - Artificial intelligence - Guidance on risk management | Published | 	https://www.iso.org/standard/77304.html
AI Management System | ISO/IEC DIS 42001 Information technology — Artificial intelligence — Management system | Published | https://www.iso.org/standard/81230.html
Biases in AI | ISO/IEC TR 24027:2021 Information technology — Artificial intelligence (AI) — Bias in AI systems and AI aided decision making | Published | https://www.iso.org/standard/77607.html
AI Performance | ISO/IEC TS 4213:2022 Information technology — Artificial intelligence — Assessment of machine learning classification performance | Published | https://www.iso.org/standard/79799.html
Ethical and societal concerns | ISO/IEC TR 24368:2022 Information technology — Artificial intelligence — Overview of ethical and societal concerns | Published | https://www.iso.org/standard/78507.html
Explainability | ISO/IEC AWI TS 6254 Information technology — Artificial intelligence — Objectives and approaches for explainability of ML models and AI systems | Under Development | https://www.iso.org/standard/82148.html
AI Sustainability | ISO/IEC AWI TR 20226 Information technology — Artificial intelligence — Environmental sustainability aspects of AI systems | Under Development | https://www.iso.org/standard/86177.html
AI Verification and Validation | ISO/IEC AWI TS 17847 Information technology — Artificial intelligence — Verification and validation analysis of AI systems | Under Development | https://www.iso.org/standard/85072.html
AI Controllabitlity | ISO/IEC CD TS 8200 Information technology — Artificial intelligence — Controllability of automated artificial intelligence systems | Published | https://www.iso.org/standard/83012.html
Biases in AI | ISO/IEC CD TS 12791 Information technology — Artificial intelligence — Treatment of unwanted bias in classification and regression machine learning tasks | Published | https://www.iso.org/standard/84110.html
AI Impact Assessment | ISO/IEC AWI 42005 Information technology — Artificial intelligence — AI system impact assessment | Under Development | https://www.iso.org/standard/44545.html
Data Quality for AI/ML | ISO/IEC DIS 5259 Artificial intelligence — Data quality for analytics and machine learning (ML) (1 to 6) | Published | https://www.iso.org/standard/81088.html
Data Lifecycle | ISO/IEC FDIS 8183 Information technology — Artificial intelligence — Data life cycle framework | Published | https://www.iso.org/standard/83002.html
Audit and Certification | ISO/IEC CD 42006 Information technology — Artificial intelligence — Requirements for bodies providing audit and certification of artificial intelligence management systems | Under Development | https://www.iso.org/standard/44546.html
Transparency | ISO/IEC AWI 12792 Information technology — Artificial intelligence — Transparency taxonomy of AI systems | Under Development | https://www.iso.org/standard/84111.html
AI Quality | ISO/IEC AWI TR 42106 Information technology — Artificial intelligence — Overview of differentiated benchmarking of AI system quality characteristics | Under Development | https://www.iso.org/standard/86903.html
Trustworthy AI | ISO/IEC TR 24028:2020 Information technology — Artificial intelligence — Overview of trustworthiness in artificial intelligence | Published | https://www.iso.org/standard/77608.html
Synthetic Data | ISO/IEC AWI TR 42103 Information technology — Artificial intelligence — Overview of synthetic data in the context of AI systems | Under Development | https://www.iso.org/standard/86899.html
AI Security | ISO/IEC AWI 27090 Cybersecurity — Artificial Intelligence — Guidance for addressing security threats and failures in artificial intelligence systems | Under Development | https://www.iso.org/standard/56581.html
AI Privacy | ISO/IEC AWI 27091 Cybersecurity and Privacy — Artificial Intelligence — Privacy protection | Under Development | https://www.iso.org/standard/56582.html
AI Governance | ISO/IEC 38507:2022 Information technology — Governance of IT — Governance implications of the use of artificial intelligence by organizations | Published | https://www.iso.org/standard/56641.html
AI Safety | ISO/IEC CD TR 5469 Artificial intelligence — Functional safety and AI systems | Published | https://www.iso.org/standard/81283.html
Beneficial AI Systems | ISO/IEC AWI TR 21221 Information technology – Artificial intelligence – Beneficial AI systems | Under Development  | https://www.iso.org/standard/86690.html

### NIST Standards

- [NIST AI Risk Management Framework](https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook)
- [NIST RMF Crosswalks](https://airc.nist.gov/AI_RMF_Knowledge_Base/Crosswalks)
- [NIST Technical and Policy Documents](https://airc.nist.gov/AI_RMF_Knowledge_Base/Technical_And_Policy_Documents)
- [NIST RMF Use Cases](https://airc.nist.gov/Usecases)
- [NIST Assessing Risks and Impacts of AI (ARIA)](https://ai-challenges.nist.gov/aria)

Additional standards can be found using the [Standards Database](https://aistandardshub.org/ai-standards-search/).

## Citing this repository

Contributors with over 50 edits can be named coauthors in the citation of visible names. Otherwise, all contributors with fewer than 50 edits are included under "et al."

### Bibtex

```
@misc{arai_repo,
  author={Josep Curto et al.},
  title={Awesome Responsible Artificial Intelligence},
  year={2025},
  note={\url{https://github.com/AthenaCore/AwesomeResponsibleAI}}
}
```

### ACM, APA, Chicago, and MLA

**ACM (Association for Computing Machinery)**

Curto, J., et al. 2025. Awesome Responsible Artificial Intelligence. GitHub. https://github.com/AthenaCore/AwesomeResponsibleAI.

**APA (American Psychological Association) 7th Edition**

Curto, J., et al. (2025). Awesome Responsible Artificial Intelligence. GitHub. https://github.com/AthenaCore/AwesomeResponsibleAI.

**Chicago Manual of Style 17th Edition**

Curto, J., et al. "Awesome Responsible Artificial Intelligence." GitHub. Last modified 2025. https://github.com/AthenaCore/AwesomeResponsibleAI.

**MLA (Modern Language Association) 9th Edition**

Curto, J., et al. "Awesome Responsible Artificial Intelligence". *GitHub*, 2025, https://github.com/AthenaCore/AwesomeResponsibleAI. Accessed 08 Jan 2025.
