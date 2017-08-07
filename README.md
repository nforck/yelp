# Predict user sentiments on the yelp data set
We want to predict, if an user comment on the yelp dataset is postive(4-5 stars)
, negative(1-2 stars) or neutral(3 stars). For this, we play around with sklearn.
## Getting Started
Please clone the repo.
Then get the file yelp_academic_dataset_review.json
from the yelp dataset to your directory.
### Prerequisites

Please have git and a docker installed in your system.


### Installing

First:
```
git clone https://github.com/nforck/yelp.git
```
Then move the dataset yelp_academic_dataset_review.json to the yelp directory and create a docker file:
```
docker build -t test .

```
