[![Awesome](awesome.svg)](https://github.com/AthenaCore/AwesomeResponsibleAI)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/AthenaCore/AwesomeResponsibleAI/graphs/commit-activity)
![GitHub](https://img.shields.io/badge/Release-PROD-yellow.svg)
![GitHub](https://img.shields.io/badge/Languages-MULTI-blue.svg)
![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)
[![GitHub](https://img.shields.io/twitter/follow/athenacoreai.svg?label=Follow)](https://twitter.com/athenacoreai)

# Awesome Responsible AI
A curated list of awesome academic research, books, code of ethics, newsletters, principles, podcast, reports, tools and regulations related to Responsible AI and Human-Centered AI.

## Contents

- [Academic Research](#academic-research)
- [Books](#books)
- [Code of Ethics](#code-of-ethics)
- [Data Sets](#data-sets)
- [Institutes](#institutes)
- [Newsletters](#newsletters)
- [Principles](#principles)
- [Podcasts](#podcasts)
- [Reports](#reports)
- [Tools](#tools)
- [Regulations](#regulations)

## Academic Research

### Challenges

- Underspecification presents challenges for credibility in modern machine learning. ([D'AMOUR, Alexander, et al., 2020](https://arxiv.org/abs/2011.03395)) `Google`

### Drift

- FreaAI: Automated extraction of data slices to test machine learning models ([Ackerman, S. et al. 2021](https://arxiv.org/pdf/2108.05620.pdf)) `IBM`
- Machine Learning Model Drift Detection Via Weak Data Slices ([Ackerman, S. et al. 2021](https://arxiv.org/pdf/2108.05319.pdf)) `IBM`


### Explainability

- Efficient Data Representation by Selecting Prototypes with Importance Weights ([Gurumoorthy et al., 2019](https://arxiv.org/abs/1707.01212)) `Amazon Development Center` `IBM Research`
- Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives ([Dhurandhar et al., 2018](https://papers.nips.cc/paper/7340-explanations-based-on-the-missing-towards-contrastive-explanations-with-pertinent-negatives)) `University of Michigan` `IBM Research`
- Contrastive Explanations Method with Monotonic Attribute Functions ([Luss et al., 2019](https://arxiv.org/abs/1905.12698))
- "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME) ([Ribeiro et al. 2016](https://arxiv.org/abs/1602.04938),  [Github](https://github.com/marcotcr/lime)) `University of Washington`
- A Unified Approach to Interpreting Model Predictions (SHAP) ([Lundberg, et al. 2017](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions),  [Github](https://github.com/slundberg/shap)) `University of Washington`
- Teaching AI to Explain its Decisions ([Hind et al., 2019](https://doi.org/10.1145/3306618.3314273)) `IBM Research`
- Boolean Decision Rules via Column Generation (Light Edition) ([Dash et al., 2018](https://papers.nips.cc/paper/7716-boolean-decision-rules-via-column-generation)) `IBM Research`
- Generalized Linear Rule Models ([Wei et al., 2019](http://proceedings.mlr.press/v97/wei19a.html)) `IBM Research`
- Improving Simple Models with Confidence Profiles ([Dhurandhar et al., 2018](https://papers.nips.cc/paper/8231-improving-simple-models-with-confidence-profiles)) `IBM Research`
- Towards Robust Interpretability with Self-Explaining Neural Networks ([Alvarez-Melis et al., 2018](https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks)) `MIT`
- Leveraging Latent Features for Local Explanations ([Luss et al., 2019](https://arxiv.org/abs/1905.12698)) `IBM Research` `University of Michigan`

### Fairness

- [LiFT: A Scalable Framework for Measuring Fairness in ML Applications](https://engineering.linkedin.com/blog/2020/lift-addressing-bias-in-large-scale-ai-applications) ([Vasuvedan et al., 2020](https://arxiv.org/abs/2008.07433)) `LinkedIn`

### Ethical Data Products

- [Building Inclusive Products Through A/B Testing](https://engineering.linkedin.com/blog/2020/building-inclusive-products-through-a-b-testing) ([Saint-Jacques et al, 2020](https://arxiv.org/pdf/2002.05819)) `LinkedIn`

### Sustainability

- Energy and policy considerations for deep learning in NLP ([Strubell, E. et al. 2019](https://arxiv.org/abs/1906.02243))
- Quantifying the carbon emissions of machine learning. ([Lacoste, A. et al. 2019](https://arxiv.org/abs/1910.09700))
- Carbon emissions and large neural network training. ([Patterson, D. et al. 2021](https://arxiv.org/abs/2104.10350)) 
- The Energy and Carbon Footprint of Training End-to-End Speech Recognizers. ([Parcollet, T., & Ravanelli, M. 2021](https://hal.archives-ouvertes.fr/hal-03190119/document))
- Sustainable AI: AI for sustainability and the sustainability of AI ([van Wynsberghe, A. 2021](https://link.springer.com/article/10.1007/s43681-021-00043-6)). AI and Ethics, 1-6
- Green Algorithms: Quantifying the carbon emissions of computation ([Lannelongue, L. et al. 2020](https://arxiv.org/abs/2007.07610))
- Machine Learning: The High Interest Credit Card of Technical Debt ([Sculley, D. et al. 2014](https://research.google/pubs/pub43146/)) `Google`

### Collections

- Google Research on Responsible AI: https://research.google/pubs/?collection=responsible-ai `Google`

## Books

### Open Access

- Interpretable Machine Learning ([Molnar, C., 2021](https://christophm.github.io/interpretable-ml-book/)) `Explainability` `Interpretability` `Transparency` `R`
- Explanatory Model Analysis ([Biecek et al., 2020](https://ema.drwhy.ai)) `Explainability` `Interpretability` `Transparency` `R`

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

## Code of Ethics

- [ACS Code of Professional Conduct](https://www.acs.org.au/content/dam/acs/rules-and-regulations/Code-of-Professional-Conduct_v2.1.pdf) by Australian ICT (Information and Communication Technology)
- [Association for Computer Machinery's Code of Ethics and Professional Conduct](https://www.acm.org/code-of-ethics)
- [IEEE Global Initiative for Ethical Considerations in Artificial Intelligence (AI) and Autonomous Systems (AS)](https://ethicsinaction.ieee.org/)
- [ISO/IEC's Standards for Artificial Intelligence](https://www.iso.org/committee/6794475/x/catalogue/)
- [Ethics guidelines for trustworthy AI](https://op.europa.eu/en/publication-detail/-/publication/d3988569-0434-11ea-8c1f-01aa75ed71a1/language-en/format-PDF/source-229277158) - European Commission document prepared by the High-Level Expert Group on Artificial Intelligence (AI HLEG).
- [Google AI Principles](https://ai.google/principles/)
- [Microsoft AI Principles](https://www.microsoft.com/en-us/ai/responsible-ai)

## Data Sets

- [An ImageNet replacement for self-supervised pretraining without humans](https://www.robots.ox.ac.uk/~vgg/research/pass/)

## Institutes

- [Open Data Institute](https://theodi.org/)
- [Ada Lovelace Institute](https://www.adalovelaceinstitute.org/)
- [The Institute for Ethical AI & Machine Learning](https://ethical.institute/)
- [Center for Responsible AI](https://airesponsibly.com/)
- [University of Oxford Institute for Ethics in AI](https://www.oxford-aiethics.ox.ac.uk/)
- [Montreal AI Ethics Institute](https://montrealethics.ai/)
- [Munich Center for Technology in Society (IEAI)](https://ieai.mcts.tum.de/)
- [Stanford University Human-Centered Artificial Intelligence (HAI)](https://hai.stanford.edu)

## Newsletters

- [Import AI](https://jack-clark.net)
- [The AI Ethics Brief](https://brief.montrealethics.ai)
- [The Machine Learning Engineer](https://ethical.institute/mle.html) 

## Principles

- [European Commission's Guidelines for Trustworthy AI](https://ec.europa.eu/futurium/en/ai-alliance-consultation)
- [IEEE's Ethically Aligned Design](https://ethicsinaction.ieee.org/)
- [The Institute for Ethical AI & Machine Learning: The Responsible Machine Learning Principles](https://ethical.institute/principles.html)

Additional:

- [FAIR Principles](https://www.go-fair.org/fair-principles/) `Findability` `Accessibility` `Interoperability` `Reuse`

## Podcasts

- [The Human-Centered AI Podcast](https://podcasts.apple.com/us/podcast/the-human-centered-ai-podcast/id1499839858)
- [Responsible AI Podcast](https://open.spotify.com/show/63Fx70r96P3ghWavisvPEQ)

## Reports

- [Inferring Concept Drift Without Labeled Data, 2021](https://concept-drift.fastforwardlabs.com) `Drift`
- [Interpretability, Fast Forward Labs, 2020](https://ff06-2020.fastforwardlabs.com) `Interpretability`
- [State of AI](https://www.stateof.ai) - from 2018 up to now -

## Tools

### Causal Inference

- [CausalNex](https://causalnex.readthedocs.io) `Python`
- [CausalImpact](https://cran.r-project.org/web/packages/CausalImpact) `R`
- [Causalinference](https://causalinferenceinpython.org) `Python`
- [dagitty](https://cran.r-project.org/web/packages/dagitty) `R`
- [DoWhy](https://github.com/Microsoft/dowhy) `Python` `Microsoft`
- [MRPC](https://cran.r-project.org/web/packages/MRPC) `R`

### Differential Privacy

- [BackPACK](https://toiaydcdyywlhzvlob.github.io/backpack) `Python`
- [DataSynthesizer: Privacy-Preserving Synthetic Datasets](https://github.com/DataResponsibly/DataSynthesizer) `Python` `Drexel University` `University of Washington`
- [diffpriv](https://github.com/brubinstein/diffpriv) `R`
- [Diffprivlib](https://github.com/IBM/differential-privacy-library) `Python` `IBM`
- [Discrete Gaussian for Differential Privacy](https://github.com/IBM/discrete-gaussian-differential-privacy) `Python` `IBM`
- [Opacus](https://opacus.ai) `Python` `Facebook`
- [PyVacy: Privacy Algorithms for PyTorch](https://github.com/ChrisWaites/pyvacy) `Python`
- [SEAL](https://github.com/Microsoft/SEAL) `Python` `Microsoft`
- [SmartNoise](https://github.com/opendp/smartnoise-core) `Python` `OpenDP`
- [Tensorflow Privacy](https://github.com/tensorflow/privacy) `Python` `Google`

### Drift

- [Alibi Detect](https://github.com/SeldonIO/alibi-detect) `Python`
- [Evidently](https://github.com/evidentlyai/evidently) `Python`
- [drifter](https://cran.r-project.org/web/packages/drifter/) `R`

### Fairness

- [Aequitas' Bias & Fairness Audit Toolkit](http://aequitas.dssg.io/) `Python`
- [AI360 Toolkit](https://github.com/Trusted-AI/AIF360) `Python` `R` `IBM`
- [Fairlearn](https://fairlearn.org) `Python` `Microsoft`
- [Fairmodels](https://fairmodels.drwhy.ai) `R`
- [fairness](https://cran.r-project.org/web/packages/fairness/) `R`
- [FairPAN - Fair Predictive Adversarial Network](https://modeloriented.github.io/FairPAN/) `R`
- [Themis ML](https://github.com/cosmicBboy/themis-ml) `Python`
- [What-If Tool](https://github.com/PAIR-code/what-if-tool) `Python` `Google`

### Interpretability/Explicability

- [AI360 Toolkit](https://github.com/Trusted-AI/AIF360) `Python` `R` `IBM`
- [breakDown: Model Agnostic Explainers for Individual Predictions](https://cran.r-project.org/web/packages/breakDown/index.html) `R`
- [ceterisParibus: Ceteris Paribus Profiles](https://cran.r-project.org/web/packages/ceterisParibus/index.html) `R`
- [DALEX: moDel Agnostic Language for Exploration and eXplanation](https://dalex.drwhy.ai) `Python` `R`
- [DALEXtra: extension for DALEX](https://modeloriented.github.io/DALEXtra) `Python` `R`
- [eli5](https://github.com/TeamHG-Memex/eli5) `Python`
- [eXplainability Toolbox](https://ethical.institute/xai.html) `Python`
- [FAT Forensics](https://fat-forensics.org/) `Python`
- [intepretML](https://interpret.ml) `Python`
- [lime: Local Interpretable Model-Agnostic Explanations](https://cran.r-project.org/web/packages/lime/index.html) `R`
- [Shapash](https://github.com/maif/shapash) `Python`
- [Skater](https://github.com/oracle/Skater) `Python` `Oracle`
- [Zennit](https://github.com/chr5tphr/zennit) `Python`

### Performance (& Automated ML)

- [automl: Deep Learning with Metaheuristic](https://cran.r-project.org/web/packages/automl/index.html) `R`
- [AutoKeras](https://github.com/keras-team/autokeras) `Python`
- [Auto-Sklearn](https://github.com/automl/auto-sklearn) `Python`
- [Featuretools](https://www.featuretools.com) `Python`
- [forester](https://modeloriented.github.io/forester/) `R`
- [NNI: Neural Network Intelligence](https://github.com/microsoft/nni) `Python` `Microsoft`
- [TPOT](http://epistasislab.github.io/tpot/) `Python`

### Responsible AI toolkit

- [Dr. Why](https://github.com/ModelOriented/DrWhy) `R` `Warsaw University of Technology`
- [Responsible AI Widgets](https://github.com/microsoft/responsible-ai-widgets) `R` `Microsoft`

### Sustainability

- [Code Carbon](https://github.com/mlco2/codecarbon) `Python`
- [Azure Sustainability Calculator](https://appsource.microsoft.com/en-us/product/power-bi/coi-sustainability.sustainability_dashboard) `Microsoft`

## Reproducible Research

- [Papers with Code](https://paperswithcode.com)
- [Papers without Code](https://www.paperswithoutcode.com)

## Regulations

- [Data Protection Laws of the Word](https://www.dlapiperdataprotection.com)

### European Union

- [General Data Protection Regulation GDPR](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32016R0679) - Legal text for the EU GDPR regulation 2016/679 of the European Parliament and of the Council of 27 April 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing Directive 95/46/EC
- [GDPR.EU Guide](https://gdpr.eu/) - A project co-funded by the Horizon 2020 Framework programme of the EU which provides a resource for organisations and individuals researching GDPR, including a library of straightforward and up-to-date information to help organisations achieve GDPR compliance ([Legal Text](https://www.govinfo.gov/content/pkg/USCODE-2012-title5/pdf/USCODE-2012-title5-partI-chap5-subchapII-sec552a.pdf)).

### United States

- State consumer privacy laws: California ([CCPA](https://www.oag.ca.gov/privacy/ccpa) and its amendment, [CPRA](https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202120220AB1490)), Virginia ([VCDPA](https://lis.virginia.gov/cgi-bin/legp604.exe?212+sum+HB2307)), and Colorado ([ColoPA](https://leg.colorado.gov/sites/default/files/documents/2021A/bills/2021a_190_rer.pdf)).
- Specific and limited privacy data laws: [HIPAA](https://www.cdc.gov/phlp/publications/topic/hipaa.html), [FCRA](https://www.ftc.gov/enforcement/statutes/fair-credit-reporting-act), [FERPA](https://www.cdc.gov/phlp/publications/topic/ferpa.html), [GLBA](https://www.ftc.gov/tips-advice/business-center/privacy-and-security/gramm-leach-bliley-act), [ECPA](https://bja.ojp.gov/program/it/privacy-civil-liberties/authorities/statutes/1285), [COPPA](https://www.ftc.gov/enforcement/rules/rulemaking-regulatory-reform-proceedings/childrens-online-privacy-protection-rule), [VPPA](https://www.law.cornell.edu/uscode/text/18/2710) and [FTC](https://www.ftc.gov/enforcement/statutes/federal-trade-commission-act).
- [EU-U.S. and Swiss-U.S. Privacy Shield Frameworks](https://www.privacyshield.gov/welcome) - The EU-U.S. and Swiss-U.S. Privacy Shield Frameworks were designed by the U.S. Department of Commerce and the European Commission and Swiss Administration to provide companies on both sides of the Atlantic with a mechanism to comply with data protection requirements when transferring personal data from the European Union and Switzerland to the United States in support of transatlantic commerce.
- [Executive Order on Maintaining American Leadership in AI](https://www.whitehouse.gov/presidential-actions/executive-order-maintaining-american-leadership-artificial-intelligence/) - Official mandate by the President of the US to 
[Privacy Act of 1974](https://www.justice.gov/opcl/privacy-act-1974) - The privacy act of 1974 which establishes a code of fair information practices that governs the collection, maintenance, use and dissemination of information about individuals that is maintained in systems of records by federal agencies.
[Privacy Protection Act of 1980](https://epic.org/privacy/ppa/) - The Privacy Protection Act of 1980 protects journalists from being required to turn over to law enforcement any work product and documentary materials, including sources, before it is disseminated to the public.
