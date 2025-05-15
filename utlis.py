{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score,RandomizedSearchCV,GridSearchCV\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures\n",
    "from sklearn.pipeline import Pipeline,FeatureUnion\n",
    "from sklearn_features.transformers import DataFrameSelector\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Lenovo\\AppData\\Local\\Temp\\ipykernel_25032\\565172087.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df['ocean_proximity'][df['ocean_proximity']==\"<1H OCEAN\"]=\"1H OCEAN\"\n"
     ]
    }
   ],
   "source": [
    "df=pd.read_csv(\"housing.csv\")\n",
    "df['ocean_proximity'][df['ocean_proximity']==\"<1H OCEAN\"]=\"1H OCEAN\"\n",
    "df['rooms_per_household' ] = df['total_rooms' ] / df['households' ]\n",
    "df['bedroms_per_rooms' ] = df['total_bedrooms' ] / df['total_rooms']\n",
    "df['population_per_household' ] = df['population' ] / df['households']\n",
    "x=df.drop(columns=[\"median_house_value\"])\n",
    "y=df['median_house_value']\n",
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,shuffle=True,random_state=42)\n",
    "num_col =x_train.select_dtypes('number').columns\n",
    "cat_col = x_train.select_dtypes(\"object\").columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_pip= Pipeline(steps=[(\"selector\",DataFrameSelector(num_col)),\n",
    "    (\"imputer\",SimpleImputer(strategy=\"median\")),\n",
    "                        (\"scaler\",StandardScaler()) ])\n",
    "cat_pip=Pipeline(steps=[(\"selector\",DataFrameSelector(cat_col)),\n",
    "    (\"imputer\",SimpleImputer(strategy=\"constant\",fill_value=\"missing\")),\n",
    "    (\"ohe\",OneHotEncoder(sparse_output=False))])\n",
    "main_pip=FeatureUnion(transformer_list=[\n",
    "    (\"num\",num_pip),\n",
    "    (\"cat\",cat_pip)]\n",
    ")\n",
    "x_train_finall= main_pip.fit_transform(x_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def new_process(x_new):\n",
    "    return main_pip.transform(x_new)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
