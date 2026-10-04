# TITANIC DATA VISUALIZATION — JUPYTER NOTEBOOK
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# The uploaded file is CSV-formatted even though its extension is .xls
df = pd.read_csv("Titanic_Dataset.xls")
df = df.rename(columns={
    "survived":"Survived","pclass":"Pclass","sex":"Sex","age":"Age",
    "sibsp":"SibSp","parch":"Parch","fare":"Fare","embarked":"Embarked"
})

# Inspection
print(df.shape)
display(df.head())
display(df.info())
display(df.describe(include="all").T)
display(df.isnull().sum())

# Derived fields
df["Survival_Status"] = df["Survived"].map({0:"Not Survived",1:"Survived"})
df["Family_Size"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
df["Age_Group"] = pd.cut(
    df["Age"], [0,12,18,35,60,np.inf],
    labels=["Child","Teen","Young Adult","Adult","Senior"],
    include_lowest=True
)

# ================= UNIVARIATE =================
# Countplot
sns.countplot(data=df, x="Survival_Status")
plt.title("Passenger Survival Distribution"); plt.show()

# Pie chart
counts=df["Survival_Status"].value_counts()
plt.pie(counts,labels=counts.index,autopct="%1.1f%%",startangle=90)
plt.title("Overall Survival Share"); plt.show()

# Countplots
for col in ["Pclass","Sex","Embarked"]:
    sns.countplot(data=df,x=col)
    plt.title(f"Passenger Count by {col}")
    plt.show()

# Histograms
sns.histplot(data=df,x="Age",bins=30,kde=True)
plt.title("Age Distribution"); plt.show()

sns.histplot(data=df,x="Fare",bins=30,kde=True)
plt.title("Fare Distribution"); plt.show()

# ================= BIVARIATE =================
sns.countplot(data=df,x="Sex",hue="Survival_Status")
plt.title("Survival Outcome by Gender"); plt.show()

gender_rate=df.groupby("Sex")["Survived"].mean()*100
sns.barplot(x=gender_rate.index,y=gender_rate.values)
plt.ylim(0,100); plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate by Gender"); plt.show()

sns.countplot(data=df,x="Pclass",hue="Survival_Status")
plt.title("Survival Outcome by Passenger Class"); plt.show()

class_rate=df.groupby("Pclass")["Survived"].mean()*100
sns.barplot(x=class_rate.index.astype(str),y=class_rate.values)
plt.ylim(0,100); plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate by Passenger Class"); plt.show()

sns.boxplot(data=df,x="Survival_Status",y="Age")
plt.title("Age Distribution by Survival"); plt.show()

sns.boxplot(data=df,x="Survival_Status",y="Fare")
plt.title("Fare Distribution by Survival"); plt.show()

sns.countplot(data=df,x="Embarked",hue="Survival_Status")
plt.title("Survival by Embarkation Port"); plt.show()

family_rate=df.groupby("Family_Size")["Survived"].mean()*100
sns.barplot(x=family_rate.index,y=family_rate.values)
plt.ylim(0,100); plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate by Family Size"); plt.show()

age_rate=df.groupby("Age_Group",observed=False)["Survived"].mean()*100
sns.barplot(x=age_rate.index.astype(str),y=age_rate.values)
plt.ylim(0,100); plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate by Age Group"); plt.show()

# ================= MULTIVARIATE =================
# Class + Gender survival heatmap
pivot=df.pivot_table(index="Pclass",columns="Sex",values="Survived",aggfunc="mean")*100
sns.heatmap(pivot,annot=True,fmt=".1f")
plt.title("Survival Rate by Class and Gender"); plt.show()

# Correlation heatmap
corr_cols=["Survived","Pclass","Age","SibSp","Parch","Fare","Family_Size"]
sns.heatmap(df[corr_cols].corr(),annot=True,fmt=".2f",center=0)
plt.title("Correlation Heatmap"); plt.show()

# Scatterplots
sns.scatterplot(data=df,x="Age",y="Fare",hue="Survival_Status",alpha=.65)
plt.title("Age vs Fare by Survival"); plt.show()

sns.scatterplot(data=df,x="Age",y="Fare",hue="Survival_Status",
                size="Pclass",alpha=.6)
plt.title("Age vs Fare with Survival and Class"); plt.show()

# Pairplot
sns.pairplot(
    df[["Survived","Pclass","Age","SibSp","Parch","Fare"]].dropna(),
    hue="Survived",diag_kind="hist",corner=True
)
plt.show()

# Jointplot
sns.jointplot(
    data=df,x="Age",y="Fare",
    hue="Survival_Status",kind="scatter"
)
plt.show()

# ================= EXTRA CHARTS =================
sns.boxplot(data=df,x="Pclass",y="Fare")
plt.title("Fare Distribution by Passenger Class"); plt.show()

sns.violinplot(data=df,x="Pclass",y="Age",hue="Sex",split=True)
plt.title("Age Distribution by Class and Gender"); plt.show()

# Pandas stacked bar
ct=pd.crosstab(df["Pclass"],df["Survival_Status"])
ct.plot(kind="bar",stacked=True,figsize=(9,5))
plt.title("Stacked Survival Count by Class")
plt.xlabel("Passenger Class"); plt.ylabel("Passengers")
plt.xticks(rotation=0); plt.show()

# Summary tables
display(df.groupby("Sex")["Survived"].agg(["count","mean"]))
display(df.groupby("Pclass")["Survived"].agg(["count","mean"]))
display(pd.crosstab(
    df["Pclass"],df["Sex"],values=df["Survived"],aggfunc="mean"
)*100)

# QUICK GUIDE:
# Countplot = How many?
# Pie chart = What share?
# Histogram = How is one numeric variable distributed?
# Barplot of rates = What percentage/probability?
# Boxplot = How do numeric distributions differ by group?
# Scatterplot = How do two numeric variables relate?
# Heatmap = How do groups/variables compare simultaneously?
# Correlation heatmap = Which numeric variables are associated?
# Pairplot = How do several numeric variables interact?
