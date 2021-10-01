# UMaaS
Urban Modelling as a Service

Tag v1.0 matches the code referenced in GISTAM 2019:

"Accelerating Urban Modelling Algorithms with Artificial Intelligence
Milton, R; Roumpani, F; (2019) Accelerating Urban Modelling Algorithms with Artificial Intelligence. In: Grueau, C and Laurini, R and Ragia, L, (eds.) Proceedings of the 5th International Conference on Geographical Information Systems Theory, Applications and Management - GISTAM. (pp. pp. 105-116). INSTICC: Heraklion, Crete, Greece."

NOTE: You MUST run databuilder.py FIRST in order to create all the data in the model-runs directory.
This will download the flow files from the Census which are needed for the trips matrices and also
build the cost matrices for the model.

Cost matrices require an intensive computation process (shortest paths run on an 8 million node network), so need to be downloaded from here:
TODO: QUANT1 matrices download from website

Later updates are to improve the neural spatial interaction model in line with the problems found in the paper.

NOTE: v1.0 references QUANT version 1, which is only for England and Wales and contains 7201 zones. After tag v1.0 we use QUANT version 2, which uses includes data for England, Wales and Scotland with 8436 zones. The data is compatible e.g. limit to 7201 out of 8436 zones and you have something that matches the data from the paper, but you need the correct zones codes file. In practice, there isn't much difference in the models produced from either.
