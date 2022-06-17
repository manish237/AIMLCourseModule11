# What drives the price of a car?

## OVERVIEW

In this application, you will explore a dataset from kaggle that contains information on 3 million used cars. Your goal is to understand what factors make a car more or less expensive. As a result of your analysis, you should provide clear recommendations to your client -- a used car dealership -- as to what consumers value in a used car.

# 1. Business understanding

## 1.1 Determine business objectives

### Business objectives
1. What consumers value in a used car. 

### Business success criteria
1. Reduce inventory of the used cars not catering to customer needs
2. Update existing inventory to help increasing car value

## 1.2 Assess situation
	
### Inventory of resources
1. Data containing information on 3 million used cars.
2. Working data analysis environment like Jupyter Notebooks
3. Individual analyzing the data.
### Requirements, assumptions, and constraints
1. Need work on the data provided. We can assume there wont be any updates/additions to the dataset
### Risks and contingencies
1. Dataset may not be latest at the time of model evaluation

## 1.3 Determine data mining goals
### Data mining goals
 1. What factors make a car more or less expensive2. 
 
### Data mining success criteria
 1. List top factors contributing to car price

## 1.4 Produce project plan
### Project plan
1. Data exploration n cleanup
2. Data transformations
3. Explore models
    1. Ridge with One Hot Encoder
    2. Lasso with One Hot Encoder
    3. SFS with One Hot Encoder
4. Compare outputs using Grid Search
5. Report back the outcome


# 2. Data understanding

## 2.1 Collect initial data
	
### Initial data collection report
1. The data was provided already in the assignment

## 2.2 Describe data
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 426880 entries, 0 to 426879
Data columns (total 18 columns):
```

| Col # | Column        | Non-Null | Count    | Dtype   |
| ----- | ------------- | -------- | -------- | ------- |
| 0     | id            | 426880   | non-null | int64   |
| 1     | region        | 426880   | non-null | object  |
| 2     | price         | 426880   | non-null | int64   |
| 3     | year          | 425675   | non-null | float64 |
| 4     | manufacturer  | 409234   | non-null | object  |
| 5     | model         | 421603   | non-null | object  |
| 6     | condition     | 252776   | non-null | object  |
| 7     | cylinders     | 249202   | non-null | object  |
| 8     | fuel          | 423867   | non-null | object  |
| 9     | odometer      | 422480   | non-null | float64 |
| 10    | title\_status | 418638   | non-null | object  |
| 11    | transmission  | 424324   | non-null | object  |
| 12    | VIN           | 265838   | non-null | object  |
| 13    | drive         | 296313   | non-null | object  |
| 14    | size          | 120519   | non-null | object  |
| 15    | type          | 334022   | non-null | object  |
| 16    | paint\_color  | 296677   | non-null | object  |
| 17    | state         | 426880   | non-null | object  |
```
dtypes: float64(2), int64(2), object(14)
memory usage: 58.6+ MB
```

## 2.3 Explore data AND 2.4 Verify data quality

| Column Name   | Value Count | Null Count | Unique Count | Constraint | Type    | Description            | Categories                                                                                                                  | Quality Issues                         |
| ------------- | ----------- | ---------- | ------------ | ---------- | ------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| id            | 426880      | 0          | 426880       | non-null   | int64   | row identifier         |                                                                                                                             | None                                   |
| region        | 426880      | 0          | 404          | non-null   | object  | region of the used car |                                                                                                                             | values have / , '                      |
| price         | 426880      | 32895      | 15655        | non-null   | int64   | price of the car       |                                                                                                                             | 0 values                               |
| year          | 425675      | 1205       | 114          | non-null   | float64 | model year             |                                                                                                                             | missing values                         |
| manufacturer  | 409234      | 17646      | 42           | non-null   | object  | manufacturer           |                                                                                                                             | missing values .. Values have spaces - |
| model         | 421603      | 5277       | 29649        | non-null   | object  | model                  |                                                                                                                             | missing values .. Values have spaces - |
| condition     | 252776      | 174104     | 6            | non-null   | object  | condition of the car   | good<br>excellent<br>like new<br>fair<br>new<br>salvage                                                                     | missing values                         |
| cylinders     | 249202      | 177678     | 8            | non-null   | object  | Number of cylinders    | 6 cylinders<br>4 cylinders<br>8 cylinders<br>5 cylinders<br>10 cylinders<br>other<br>3 cylinders<br>12 cylinders            | missing values .. Values have spaces   |
| fuel          | 423867      | 3013       | 5            | non-null   | object  | Fuel type              | gas<br>other<br>diesel<br>hybrid<br>electric                                                                                | missing values                         |
| odometer      | 422480      | 4400       | 104870       | non-null   | float64 | Odometer reading       |                                                                                                                             | missing values                         |
| title\_status | 418638      | 8242       | 6            | non-null   | object  | title status           | clean<br>rebuilt<br>salvage<br>lien<br>missing<br>parts only                                                                | missing values .. Values have spaces   |
| transmission  | 424324      | 2556       | 3            | non-null   | object  | transmission type      | automatic<br>other<br>manual                                                                                                | missing values                         |
| VIN           | 265838      | 161042     | 118246       | non-null   | object  | VIN                    |                                                                                                                             | missing values                         |
| drive         | 296313      | 130567     | 3            | non-null   | object  | drive type             | 4wd<br>fwd<br>rwd                                                                                                           | missing values                         |
| size          | 120519      | 306361     | 4            | non-null   | object  | size                   | full-size<br>mid-size<br>compact<br>sub-compact                                                                             | missing values .. Values have -        |
| type          | 334022      | 92858      | 13           | non-null   | object  | type                   | sedan<br>SUV<br>pickup<br>truck<br>other<br>coupe<br>hatchback<br>wagon<br>van<br>convertible<br>mini-van<br>off-road<br>bus | missing values .. Values have -        |
| paint\_color  | 296677      | 130203     | 12           | non-null   | object  | color                  | white<br>black<br>silver<br>blue<br>red<br>grey<br>green<br>custom<br>brown<br>yellow<br>orange<br>purple                   | missing values                         |
| state         | 426880      | 0          | 51           | non-null   | object  | location / state       |                                                                                                                             | None                                   |

# 3. Data preparation

## 3.1 Select data
Data was loaded into a dataframe for further cleanup and exploration.
The cleanup was done as per below
1. **Manufacturer and model** - Missing values for *manufacturer* field was derived by *model* name (available) and online lookup of model name and manufacturer information. Rows with undetermined manufacturers and models were dropped.
2. **VIN** - Rows with invalid *VIN* numbers (having VINS less than 17 characters) were dropped.
3. **year** - Missing *year* values were  derived from *VIN* decoding 
4. **fuel** - Missing *fuel* values were derived from *model* column and rest were defaulted to top contributing value.
5. **transmission** - Missing *transmission* values were derived from *model* column and rest were defaulted to top contributing value.
6. **cylinder** - Missing *cylinder* values were derived from *model* column and rest were defaulted to top contributing value based on *year*.
7. **drive** - Missing *drive* values were derived from *model* and *cylinders* columns and rest were defaulted to top contributing value.
8. **type** - Missing *type* values were derived from *model* column and rest were dropped.
9. **size** - Missing *size* values were derived from *model* and *cylinders* columns.
10. **condition** - Missing *condition* values were defaulted to mean value.
11. **title_status** -  Missing *title_status* values were defaulted to mean value.
12. **odometer** -  Missing *odometer* values were defaulted to mean value.

	
### Rationale for inclusion/exclusion
**VIN** n **id** columns were dropped from the dataset as they seems to be row or vehicle identifiers which should not be impacting the price.

1. **model** was dropped as the column values have lot of redundant information.
2. **size** was dropped as the column is redundant w.r.t type.
3. **odometer** dropped after running permutation_importance analysis using ridge regression.
4. **year** dropped after running permutation_importance analysis using ridge regression.
5. **transmission** dropped after running permutation_importance analysis using ridge regression.
6. **title_status** dropped after running permutation_importance analysis using ridge regression.

## 3.2 Clean data
	
### Data cleaning report
Various special characters from different string columns were cleaned up to work well with data transformation tools.

## 3.3 Construct Data

### Generated Records
**region** column was replaced by US regions based on *state* column

# 4. Modeling

	
## 4.1 Select modeling technique

### Modeling technique

Following models were trained
1. Ridge Regression with One Hot Encoding
2. Cross Validation of Ridge Regression with One Hot Encoding
3. Linear Regression with One Hot Encoding
4. Bayesian Regression with One Hot Encoding
5. Random Forest Regression with One Hot Encoding

# 5. Evaluation

## 5.1 Evaluate results

### Approved models
After comparing different model run on test data, Linear regression gave least MSE for test data identifying following top 5 features
1. manufacturer
2. type
3. cylinders
4. fuel
5. condition

Second best performing model was bayesian regression identifying following top 5 features
1. fuel
2. manufacturer
3. type
4. condition
5. drive

## 5.2 Determine next steps

### List of possible actions

Review the data clean up strategy and run other regression models to see if we can find another better performing model.


# 6. Deployment

## 6.1 Plan deployment

### Deployment plan

## 6.2 Plan monitoring and maintenance

### Monitoring and maintenance plan

## 6.3 Produce final report
	
This README file will act as final report