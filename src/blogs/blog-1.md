# Gender of Spanish nouns

**Date:** 18/1/2024

<!-- - [How I got here?](#how-i-got-here)
- [A data-driven approach](#a-data-driven-approach)
  - [Data](#data)&nbsp;&nbsp;|&nbsp;&nbsp;[Exploratory data analysis](#exploratory-data-analysis)&nbsp;&nbsp;|&nbsp;&nbsp;[Sanity checks](#sanity-checks)&nbsp;&nbsp;|&nbsp;&nbsp;[Key findings from analysis](#key-findings-from-analysis)
- [Creating a machine learrning model](#creating-a-machine-learning-model)
  - [Data (n=1813)](#data-n1813)&nbsp;&nbsp;|&nbsp;&nbsp;[Reduced data (n=180)](#reduced-data-n180)
- [Final set of rules and exceptions](#final-set-of-rules-and-exceptions)
  - [Exceptions as a plot](#exceptions-as-a-plot)
- [Future directions](#future-directions) -->

## How I got here?

I am currently spending a gap year in Spain, and as part of my daily routine I have been following some exercises to practice my Spanish as outlined in this [website](https://studyspanish.com/).

<p align="center">
  <img src="/assets/blog-1/Untitled.png" width="500" />
</p>

It is fairly known for those learning Spanish that nouns are gendered. Identifying the gender of a noun is important, and it establishes how we change the article of noun in a sentence i.e., el chico, los chicos, la chica, las chicas. Most tutors and online resources will claim that learning which gender a noun holds is mainly a memory game with only a few rules to guide you. These rules however have exceptions.

**Some of these rules are:**

- Most nouns ending in -o will be masculine.
- Most nouns ending in -a will be feminine.
- All words ending in -ción and -sión will be feminine.

This is evidenced above with the words chico and chica. Then we get to the noun **jardín** (garden) and straight away I have some questions. Why is jardín masculine? Are there any general rules I can follow with words with similar endings?

The first thing I do is ask Bard. Here is what it told me.

<p align="center">
  <img src="/assets/blog-1/Untitled%201.png" width="1000" />
</p>

Apparently jardín stems from the Latin *hortus*. Does this mean we can infer gender purely based on the Latin origins of Spanish nouns?

<p align="center">
  <img src="/assets/blog-1/Untitled%202.png" width="1000" />
</p>

Now this is interesting - if this is true then having an understanding on whether the Latin translation of a Spanish noun belongs to the second declension can determine the gender. And apparently there are NO exceptions to this rule. But clearly learning Latin grammar to help me with Spanish grammar probably isn’t the quickest way to learn the gender of all nouns, but it’s nice to know that that option exists. Perhaps we can learn about patterns of nouns in the second declension - let’s continue prompting Bard.

<p align="center">
  <img src="/assets/blog-1/Untitled%203.png" width="1000" />
</p>

Point 2 could potentially be a pretty useful rule when we are left with nouns with endings other than -o, -a, -ción and -sión. Let’s sanity check this using Bard.

<p align="center">
  <img src="/assets/blog-1/Untitled%204.png" width="1000" />
</p>

<p align="center">
  <img src="/assets/blog-1/Untitled%205.png" width="1000" />
</p>

Well, looks like that was a massive failure. Let’s try with ChatGPT.

<p align="center">
  <img src="/assets/blog-1/Untitled%206.png" width="800" />
</p>

Another swing and a miss for LLMs. One would think this is a task that AI could help us with but apparently not. This means we can’t answer my previous question of “Are there any general rules I can follow with words with similar endings?”. How can we answer this? Let’s turn to traditional machine learning.

## A data-driven approach

The idea here would be to get some data of Spanish nouns with their corresponding gender and identify patterns in noun endings - we can do this through summary statistics and visuals. From here, we could build a binary classifier taking in some engineered features of the noun endings and see how well the classifier can establish rules. A decision-tree would be the most suitable choice here. So let’s find some data.

### Data

Unfortunately a structured dataset of Spanish nouns with their gender doesn’t exist, at least for free. There are some APIs out there like [Merriam Webster](https://dictionaryapi.com/products/api-spanish-dictionary), but these have limitations that would turn this evening analysis into a week or two-week project. Through some googling, I found a [website](https://frequencylists.blogspot.com/2015/12/the-2000-most-frequently-used-spanish.html) that lists the top 2000 most frequently used Spanish nouns.

<p align="center">
  <img src="/assets/blog-1/Untitled%207.png" width="600" />
</p>

No sources or information about how this list was formed, but hey it’s good enough. Luckily the text follows somewhat of a structure, so with some quick cleaning we can make this into a ML-ready dataset. First we copied the text directly from the website into a .txt format. We read the .txt file into a Jupyter notebook and separated the string by a delimiter into columns. This allowed us to easily identify any “problematic” rows.

Most of the data looked like what we can see above, but some irregular patterns existed.

- Some lines consisted of multiple Spanish translations for the english noun. We created new distinct lines for these cases.
- Some Spanish translations were more than one word. We omit these as we really only want to focus on one word data. From memory, there may have been five to ten cases.
- Some translations had extra information to tell the reader whether the noun was plural. We left these plural nouns i, however in their singular form.
- Some Spanish nouns have two forms for masculine and feminine. We created two distinct lines for these cases. For example:
    - [ORIGINAL FORMAT] **author** - autor/a - masculine/feminine
    - [NEW FORMAT] **author** - autor - masculine
    - [NEW FORMAT] **author** - autora - feminine
    

As there were not many cases that needed cleaning this was done manually. We have provided the resultant dataset here.

[spanish-nouns.txt](spanish-nouns.txt)

We provide a snapshot of the data, alongside some code to read and apply some final cleaning steps.

<details>
<summary>Click to expand Python code</summary>

```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# import text file extracted from...
# https://frequencylists.blogspot.com/2015/12/the-2000-most-frequently-used-spanish.html
df = pd.read_csv('spanish-nouns.txt', sep='\t', header=None)

# split string into columns by space delimiter
df = df[0].str.split(' ', expand=True)

# clean words with +1 words for spanish translation
df = df[df[3].str.contains('-')]

# remove extra columns
df = df[[0,1,2,4]]

# remove duplicate rows based on df[2] column, keep first
df = df.drop_duplicates(subset=[2], keep='first')

# remove last character from df[1] column and strip whitespace
df[1] = df[1].str[:-1].str.strip()

# rename columns
df.columns = ['original-id', 'english', 'spanish', 'gender']

# rename value in df['gender'] for another value
df['gender'] = np.where(df['gender'] == 'masculine/feminine', 'both', df['gender'])

df.head()
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%208.png" width="500" />
</p>

### Exploratory data analysis

It should be noted, because we removed some of the data in the manual cleaning and we removed duplicates, we were left with 1813 nouns. Duplicates in this context looked like this:

- hit - golpe - masculine
- blow - golpe - masculine
- knock - golpe - masculine
- bump - golpe - masculine
- stroke - golpe - masculine

The Spanish noun is the same gender for each english translation, therefore we remove duplicates and keep first instance.

Next we look into the distribution of genders for the given set of Spanish nouns through some pie-charts.

<details>
<summary>Click to expand Python code</summary>

```python
## Gender Proportion
df_gender=pd.DataFrame(dict(Counter(df["gender"])).items(),
                              columns=["Gender","Frequency"])

# explosion
explode = (0.05, 0.05, 0.05)
  
# Pie Chart
plt.pie(df_gender['Frequency'], labels=['masculine','feminine','both'],
        autopct='%1.1f%%', pctdistance=0.85,
        explode=explode)
  
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
  
# Adding Title of chart
plt.title('Gender of most frequently used Spanish nouns')
  
# Displaying Chart
plt.show()

# without both
# explosion
explode = (0.05, 0.05)
  
# Pie Chart
plt.pie(df_gender[df_gender['Gender']!='both']['Frequency'], labels=['masculine','feminine'],
        autopct='%1.1f%%', pctdistance=0.85, colors=['tab:blue','tab:orange'],
        explode=explode)

# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
  
# Adding Title of chart
plt.title('Gender of most frequently used Spanish nouns')
  
# Displaying Chart
plt.show()
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%209.png" width="500" />
</p>

<p align="center">
  <img src="/assets/blog-1/Untitled%2010.png" width="500" />
</p>

The gender column consists of three categories: masculine, feminine and masculine/feminine. The latter category are nouns that do not change spelling to accomodate a different gender. They make up 2% of the data (n=37). For your reference, here are the nouns (in order of frequency rank).

'bebé', 'orden', 'otros', 'idiota', 'agente', 'detective', 'teniente', 'juez', 'arte', 'paciente', 'guardia', 'sargento', 'testigo', 'doble', 'comandante', 'cliente', 'estudiante', 'gerente', 'criminal', 'amante', 'artista', 'asistente', 'profesional', 'gigante', 'regular', 'piloto', 'espía', 'cobarde', 'cantante', 'colega', 'rehén', 'terapeuta', 'nazi', 'dentista', 'dependiente', 'mortal', 'psiquiatra'

When we focus on solely the two genders, we find that there are slightly more masculine words (54.5%) than feminine words (45.5%) in the dataset. Let’s check whether this distribution changes if we remove words that follow some of the aforementioned rules.

<details>
<summary>Click to expand Python code</summary>

```python
# only consider words that don't end in 'o' or 'a' or ción or sión
cond = (df['gender']!='both') & (~df['spanish'].str.endswith('ción')) &\
        (~df['spanish'].str.endswith('sión')) & (~df['spanish'].str.endswith('o')) &\
        (~df['spanish'].str.endswith('a'))

df_gender_rest =pd.DataFrame(dict(Counter(df[cond]["gender"])).items(),
                              columns=["Gender","Frequency"])

explode = (0.05, 0.05)

# Pie Chart
plt.pie(df_gender_rest['Frequency'], labels=['masculine','feminine'],
        autopct='%1.1f%%', pctdistance=0.85, colors=['tab:blue','tab:orange'],
        explode=explode)

# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
  
# Adding Title of chart
plt.title('Gender of most frequently used Spanish nouns \n (excluding words ending in -o or -a or -ción or -sión)')
  
# Displaying Chart
plt.show()
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2011.png" width="500" />
</p>

Interesting! A relatively crude rule could be if the words don’t end with 'o' or 'a' or ción or sión, then assume a masculine gender. You’ll only get 109 nouns wrong, not to mention the exceptions to the rules in the condition i.e., some masculine words end with -a and some feminine words end with -o.

> *A relatively crude rule could be if the words don’t end with 'o' or 'a' or ción or sión, then assume a masculine gender. You have a 77% chance of being correct.*
> 

Something to note here is the drop in the sample size of nouns (n=473) when we remove popular nouns with these endings. This tells us that a majority of nouns (73%) do end with 'o' or 'a' or ción or sión. So let’s explore how correct these established rules are.

### Sanity checks

First let’s create some subsets of the data to work with. We first remove the cases were the noun can take both genders (n=37), which leaves us with 1776 nouns.

<details>
<summary>Click to expand Python code</summary>

```python
# focus on data without both
df_noboth = df[df.gender != 'both'][['english', 'spanish', 'gender']].copy()

# create dataframe of words ending in 'a'
df_a = df_noboth[df_noboth['spanish'].str.endswith('a')]

# create dataframe of words ending in 'o'
df_o = df_noboth[df_noboth['spanish'].str.endswith('o')]

# create dataframe of words not ending in 'a' or 'o'
df_ción_sión = df_noboth[(df_noboth['spanish'].str.endswith('ción')) | (df_noboth['spanish'].str.endswith('sión'))]
```

</details>

**Sanity check #1 - nouns that end with -a are feminine**

We find that this rule is correct 96% of the time - which is surprisingly high! Leaving us with 24 exceptions in out dataset (see below).

<details>
<summary>Click to expand Python code</summary>

```python
print(df_a['gender'].value_counts())
print('\n', df_a['gender'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# Exceptions
display(df_a[df_a['gender'] == 'masculine'].drop('gender', axis=1).iloc[0:12].T)
display(df_a[df_a['gender'] == 'masculine'].drop('gender', axis=1).iloc[12:].T)
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2012.png" width="350" />
</p>

Exceptions:

<p align="center">
  <img src="/assets/blog-1/Untitled%2013.png" width="1000" />
</p>


**Sanity check #2 - nouns that end with -o are masculine**

We find that this rule is even more true! 99.7% of nouns with an endings of -a are feminine. Leaving us with only 2 exceptions in out dataset (see below).

<details>
<summary>Click to expand Python code</summary>

```python
print(df_o['gender'].value_counts())
print('\n', df_o['gender'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# Exceptions
display(df_o[df_o['gender'] == 'feminine'].drop('gender', axis=1).T)
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2014.png" width="350" />
</p>

Exceptions:

<p align="center">
  <img src="/assets/blog-1/Untitled%2015.png" width="350" />
</p>

**Sanity check #3 - nouns that end with -ción or sión are feminine**

We find that this rule is correct 100% of the time.

<details>
<summary>Click to expand Python code</summary>

```python
print(df_ción_sión['gender'].value_counts())
print('\n', df_ción_sión['gender'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2016.png" width="350" />
</p>

We found some more general rules [online](https://www.drlemon.com/Grammar/gender.html):

**Masculine rules**
- All words ending with -aje
- Most words ending with r, l, s, n

**Feminine rules**
- words ending with -tad, -dad, -tud

Let’s check the validity of these too, however I generalise these rules a bit just to see if there are any higher level rules. For example, the feminine rules all end with -d, so the rule will check all words ending with -d instead. Another example of this is with the masculine rule that most nouns end with -n. We know this can’t be true as this conflicts with the ción and sión rule.

<details>
<summary>Click to expand Python code</summary>

```python
df_aje = df_noboth[df_noboth['spanish'].str.endswith('aje')]
df_r = df_noboth[df_noboth['spanish'].str.endswith('r')]
df_l = df_noboth[df_noboth['spanish'].str.endswith('l')]
df_s = df_noboth[df_noboth['spanish'].str.endswith('s')]
df_n = df_noboth[(df_noboth['spanish'].str.endswith('n')) & (~df_noboth['spanish'].str.endswith('ción')) &\
    (~df_noboth['spanish'].str.endswith('sión'))]
df_d = df_noboth[df_noboth['spanish'].str.endswith('d')]
```

</details>

**Sanity check #4 - nouns that end with -aje are masculine**

We find that this rule is correct 100% of the time. Perfect. However there aren’t many words with this ending (n=13). We list these below.

<details>
<summary>Click to expand Python code</summary>

```python
print(df_aje['spanish'].iloc[0:10].tolist()); print(df_aje['spanish'].iloc[10:].tolist())
print('\n', df_aje['gender'].value_counts())
print('\n', df_aje['gender'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2017.png" width="1000" />
</p>

**Sanity check #5 - nouns that end with -r are masculine**

We find that this rule is correct 97.7% of the time - which is also quite high. Leaving us with only two exceptions - mujer (woman) and flor (flower).

<details>
<summary>Click to expand Python code</summary>

```python
# words ending in 'r'
print(df_r['spanish'].iloc[0:10].tolist()); print(df_r['spanish'].iloc[10:20].tolist())
print(df_r['spanish'].iloc[20:30].tolist()); print(df_r['spanish'].iloc[30:40].tolist())
print(df_r['spanish'].iloc[40:50].tolist()); print(df_r['spanish'].iloc[50:60].tolist())
print(df_r['spanish'].iloc[60:70].tolist()); print(df_r['spanish'].iloc[70:82].tolist())

print('\n', df_r['gender'].value_counts())
print('\n', df_r['gender'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# Exceptions
print('\n', 'Exceptions (feminine):')
display(df_r[df_r['gender'] == 'feminine'].drop('gender', axis=1).T)
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2018.png" width="1000" />
</p>

**Sanity check #6 - nouns that end with -s are masculine**

We find that this rule is correct 95% of the time. Leaving us with only one exception - **crisis**. However there aren’t many words with this ending if only consider non-plural nouns (n=20).

<details>
<summary>Click to expand Python code</summary>

```python
# words ending in 's' - consider plural words
print(df_s['spanish'].iloc[0:10].tolist()); print(df_s['spanish'].iloc[10:20].tolist())

print('\n', df_s['gender'].value_counts())
print('\n', df_s['gender'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# Exceptions
print('\n', 'Exceptions (feminine):')
display(df_s[df_s['gender'] == 'feminine'].drop('gender', axis=1).T)
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2019.png" width="1000" />
</p>


**Sanity check #7 - nouns that end with -l are masculine**

We find that this rule is correct 92.2% of the time. Leaving us with only four exceptions: **señal** (sign), **cárcel** (jail), **piel** (skin) and **sal** (salt).

<details>
<summary>Click to expand Python code</summary>

```python
# words ending in 'l'
print(df_l['spanish'].iloc[0:10].tolist()); print(df_l['spanish'].iloc[10:20].tolist())
print(df_l['spanish'].iloc[20:30].tolist()); print(df_l['spanish'].iloc[30:40].tolist())
print(df_l['spanish'].iloc[40:51].tolist())

print('\n', df_l['gender'].value_counts())
print('\n', df_l['gender'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# Exceptions
print('\n', 'Exceptions (feminine):')
display(df_l[df_l['gender'] == 'feminine'].drop('gender', axis=1).T)
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2020.png" width="1000" />
</p>


**Sanity check #8 - nouns that end with -n are masculine (excluding -sión and -ción)**

We find that this rule is correct 88.4% of the time - a bit lower. Leaving us with eight exceptions (see below).

<details>
<summary>Click to expand Python code</summary>

```python
# words ending in 'n' except ción or sión
print(df_n['spanish'].iloc[0:10].tolist()); print(df_n['spanish'].iloc[10:20].tolist())
print(df_n['spanish'].iloc[20:30].tolist()); print(df_n['spanish'].iloc[30:40].tolist())
print(df_n['spanish'].iloc[40:50].tolist()); print(df_n['spanish'].iloc[50:60].tolist())
print(df_n['spanish'].iloc[60:].tolist())

print('\n', df_n['gender'].value_counts())
print('\n', df_n['gender'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# Exceptions
print('\n', 'Exceptions (feminine):')
display(df_n[df_n['gender'] == 'feminine'].drop('gender', axis=1).T)
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2021.png" width="1000" />
</p>

**Sanity check #9 - nouns that end with -d are feminine**

We find that this rule is correct 94.4% of the time - supporting the  extension of the -tad, -dad, -tud rule. Leaving us with only three exceptions: **récord** (record), **césped** (grass) and **ataúd** (coffin).

<details>
<summary>Click to expand Python code</summary>

```python
# words ending in 'd'
print(df_d['spanish'].iloc[0:10].tolist()); print(df_d['spanish'].iloc[10:20].tolist())
print(df_d['spanish'].iloc[20:30].tolist()); print(df_d['spanish'].iloc[30:40].tolist())
print(df_d['spanish'].iloc[40:50].tolist()); print(df_d['spanish'].iloc[50:].tolist())

print('\n', df_d['gender'].value_counts())
print('\n', df_d['gender'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# Exceptions
print('\n', 'Exceptions (masculine):')
display(df_d[df_d['gender'] == 'masculine'].drop('gender', axis=1).T)
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2022.png" width="1000" />
</p>

### Key findings from analysis

- Following the established nine rules, we cover 1596 nouns (90%) of the dataset.
- Among these nouns, the rules will guide you to 1556 correct guesses and 44 wrong guesses.
- The rule with the most exceptions is that nouns that end with -a are feminine with 24 incorrect guesses.
- If we think of wrong guesses as single rules (i.e., mujer is feminine, etc.) we could say that altogether there are 53 rules (9 + 44) for the 1596 nouns.
- If we consider the entire dataset, and treat each noun not covered by the 9 rules as a single rule, then we have in total 233 rules (53 + 180).

## Creating a machine learning model

Considering there are still 233 bits of information to still remember, let’s explore whether a binary classifier can help identify any other rules to add to the 9. As a starting point we use the last letter/s to engineer one-hot-encoded features. There are four groups of features: 

- last letter
- last two letters (as one token)
- last three letter
- last four letters.

We use a decision tree due to its explainability. We can extract the rules of the tree and assess which rules are most informative. The only parameter we tweak is `max_depth` to stop the tree from growing too large. We also train and test on the same data, because we want to use all the data to create the decision tree.

### Data (n=1813)

We start with all the data.

<details>
<summary>Click to expand Python code</summary>

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
import graphviz

# Assuming your data is stored in a DataFrame called 'df_noboth'
# with columns 'spanish' and 'gender'

df_noboth['last-letter'] = df_noboth['spanish'].apply(lambda x: x[-1])
df_noboth['last-letters-second-last'] = df_noboth['spanish'].apply(lambda x: x[-2:])
df_noboth['last-letters-third-last'] = df_noboth['spanish'].apply(lambda x: x[-3:])
df_noboth['last-letters-fourth-last'] = df_noboth['spanish'].apply(lambda x: x[-4:])

# Step 2: Feature Extraction
X_last = df_noboth['last-letter'].values.reshape(-1, 1)
X_second_last = df_noboth['last-letters-second-last'].values.reshape(-1, 1)
X_third_last = df_noboth['last-letters-third-last'].values.reshape(-1, 1)
X_fourth_last = df_noboth['last-letters-fourth-last'].values.reshape(-1, 1)

# Step 3: One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
feature_matrices = []
feature_names = []

for feature in ['last-letter', 'last-letters-second-last', 'last-letters-third-last', 'last-letters-fourth-last']:
    # Original feature
    X_feature = df_noboth[feature].values.reshape(-1, 1)
    X_feature_encoded = encoder.fit_transform(X_feature)
    feature_matrices.append(X_feature_encoded)
    feature_names.extend([f'{feature}_{value}' for value in encoder.get_feature_names_out()])

# Combine the feature matrices
X_combined = np.hstack(feature_matrices)

y = df_noboth['gender'].map({'masculine': 0, 'feminine': 1})

# Step 5: Choose a Classifier
classifier = DecisionTreeClassifier(random_state=42, max_depth=5)

# Step 6: Train the Model
classifier.fit(X_combined, y)

# Step 7: Evaluate the Model
y_pred = classifier.predict(X_combined)

print("Accuracy:", accuracy_score(y, y_pred))
print("Classification Report:\n", classification_report(y, y_pred))

# Step 8: Visualize the Decision Tree
dot_data = export_graphviz(classifier, out_file=None, feature_names=feature_names, class_names=['masculine', 'feminine'],
                           filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Saves the decision tree visualization as a PDF or PNG file
graph.view("decision_tree")  # Opens the decision tree visualization in the default viewer
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2023.png" width="500" />
</p>

<p align="center">
  <img src="/assets/blog-1/decision_tree.png" width="1200" />
</p>

Let’s review the resultant decision tree. 

- The root node checks to see whether the noun ends with -a.
- From here, if this is true, the tree continues to identify nouns that end with -ema and -rema. All nouns that end with -ema (**problema**, **sistema**, **tema**, **poema**, **esquema**), except **crema** are masculine.
- The other path, if this is true, focuses on whether the noun ends with -día (**día**, **melodía**, **mediodía**). **melodía** is feminine and the rest are exceptions to the -a rule - not too informative. The rest of the rules are quite specific.
- If we take the first -ema and -rema rule, we go from 233 rules to 229 rules - not a huge change.
- Let’s look at the other path, if the noun doesn’t end with -a. The first node in this path checks to see whether the noun ends with -ión (127 nouns).
- Continuing down this path tells us that we could modify the -sión or -ción, to a -ión rule with three exceptions: **camión** (truck)**, anfitrión** (host) and **avión** (plane). By doing this, we could reduce the masculine -n rule which contains eight exceptions, four of which end with -ión (**opinión**, **conexión**, **religión**, **región**). This would change the count from 233 rules to 229.
- For nouns that don’t end with -ión, the next check is nouns ending with -d and then -e. There is not much information to gain with nouns ending in -d because there are three exceptions.
- For nouns ending with -e, we find that 111 out of 142 nouns are masculine. This pseudo-rule achieves a 78% accuracy. Therefore, this leaves us with 32 rules (1 + 31) and in total reduces the 233 rules to 133. This reduces our number of rules by 43%.

### Reduced data (n=180)

Let’s retrain the model on a smaller subset of the data, considering we are happy with learning some exceptions as rules. The subset will omit nouns that meet the following rules and their exceptions.

- nouns ending in -o will be masculine (**rule count: 1**)
- exceptions of the -o rule: mano and radio (**rule count: 3**)
- nouns ending in -a will be feminine (**rule count: 4**)
- exceptions of the -a rule: 24 nouns (**rule count: 28**)
- all words ending in -ción and sión will be feminine (**rule count: 29**)
- nouns that end with -aje are masculine (**rule count: 30**)
- nouns that end with -r are masculine (**rule count: 31**)
- exceptions of the -r rule: mujer and flor (**rule count: 33**)
- nouns that end with -s are masculine (**rule count: 34**)
- exceptions of the -s rule: crisis (**rule count: 35**)
- nouns that end with -l are masculine (**rule count: 36**)
- exceptions of the -l rule: 4 nouns (**rule count: 40**)
- nouns that end with -n are masculine excluding -sión and -ción (**rule count: 41**)
- exceptions of the -n rule: 8 nouns (**rule count: 49**)
- nouns that end with -d are feminine (**rule count: 50**)
- exceptions of the -d rule: césped, récord and ataúd (**rule count: 53**)

In total, 9 rules and 44 exceptions.

<details>
<summary>Click to expand Python code</summary>

```python
# new df that remove all nouns in other subsets i.e., df_n, df_s, df_l, df_r, df_aje, df_ción_sión, df_o, df_a
df_reduced = df_noboth[~df_noboth['spanish'].isin(df_n['spanish'].tolist() + df_s['spanish'].tolist() +\
    df_l['spanish'].tolist() + df_r['spanish'].tolist() + df_aje['spanish'].tolist() + df_d['spanish'].tolist()  +\
    df_ción_sión['spanish'].tolist() + df_o['spanish'].tolist() + df_a['spanish'].tolist())].copy()

df_reduced['last-letters-fifth-last'] = df_reduced['spanish'].apply(lambda x: x[-5:])

# Step 2: Feature Extraction
X_last = df_reduced['last-letter'].values.reshape(-1, 1)
X_second_last = df_reduced['last-letters-second-last'].values.reshape(-1, 1)
X_third_last = df_reduced['last-letters-third-last'].values.reshape(-1, 1)
X_fourth_last = df_reduced['last-letters-fourth-last'].values.reshape(-1, 1)
X_fifth_last = df_reduced['last-letters-fifth-last'].values.reshape(-1, 1)

# Step 3: One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
feature_matrices = []
feature_names = []

for feature in ['last-letter', 'last-letters-second-last', 'last-letters-third-last', 'last-letters-fourth-last', 'last-letters-fifth-last']:
    # Original feature
    X_feature = df_reduced[feature].values.reshape(-1, 1)
    X_feature_encoded = encoder.fit_transform(X_feature)
    feature_matrices.append(X_feature_encoded)
    feature_names.extend([f'{feature}_{value}' for value in encoder.get_feature_names_out()])

# Combine the feature matrices
X_combined = np.hstack(feature_matrices)

y = df_reduced['gender'].map({'masculine': 0, 'feminine': 1})

# Step 5: Choose a Classifier
classifier = DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=6)

# Step 6: Train the Model
classifier.fit(X_combined, y)

# Step 7: Evaluate the Model
y_pred = classifier.predict(X_combined)

print("Accuracy:", accuracy_score(y, y_pred))
print("Classification Report:\n", classification_report(y, y_pred))

# Step 8: Visualize the Decision Tree
dot_data = export_graphviz(classifier, out_file=None, feature_names=feature_names, class_names=['masculine', 'feminine'],
                           filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)

# Saves the decision tree visualization as a PNG file
graph.render("decision_tree2", format='png')
graph.view("decision_tree2")  # Opens the decision tree visualization in the default viewer
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2024.png" width="500" />
</p>

<p align="center">
  <img src="/assets/blog-1/decision_tree2.png" width="500" />
</p>

For this model, we add one extra parameter to prevent excessive splitting to encourage more generalised rules using the `min_samples_leaf` parameter, but we also increase the max depth to increase the scope. Also, we add one extra feature group where we consider the last five letter - purely for experimental purposes. 

The first evident observation is the drop in performance with an accuracy of 77%. This is probably due to the class imbalance where we are lacking a lot of feminine cases (f1 score for feminine class = 0.37), and it also seems that the model is finding it harder to identify informative splits. Nevertheless there are some useful rules here.

- The root node checks to see whether the noun ends with -z. It appears that more feminine cases end in -z but its fairly even (seven masculine and nine feminine).
    - Upon further analysis, two rules could be established here. If noun ends in -uz or -iz, then its feminine, otherwise noun ending in -z is masculine. This leaves over four feminine exceptions: **voz**, **paz**, **nuez** and **raíz**. So you go from 16 rules to 6 rules.
- From here, we follow to the node checking if last letter is -e. Here we find that if it doesn’t then there are 35 nouns left over. 32 of which are masculine. The exceptions are **mamá** (mum), **ley** (law), **tribu** (tribe). This node helps us establish 4 rules.
- For nouns that do end with -e, the first split we observe is whether the noun ends with -que. We find that all nouns that in -que are masculine. New rule!
- From here, we get a series of new rules for specific endings:
    - nouns that end with -orte are masculine (6 nouns)
    - nouns that end with -ante are masculine (6 nouns)
    - nouns that end with -le are masculine (9 nouns) with one exception (**calle**).
- After theses splits, there remains 97 nouns. 67 of which are masculine which if this were a rule it would achieve a 69% accuracy.
- Alternatively, we could scrap all the specific -e rules from above and establish a generalised -e rule which would be a 98/31 split. If we assume all are masculine, then we can turn this from 129 rules to 32 rules (1 rule + 31 exceptions). This can also be merged with the previous -aje rule.

**Here are the 31 feminine exceptions for a generalised (masculine) -e rule.**

'gente', 'noche', 'madre', 'mente', 'muerte', 'parte', 'sangre', 'suerte', 'calle', 'clase', 'llave', 'tarde', 'fe', 'leche', 'carne', 'base', 'nieve', 'madame', 'medianoche', 'fuente', 'superficie', 'torre', 'serpiente', 'serie', 'corriente', 'especie', 'fiebre', 'nube', 'fase', 'hambre', 'suite'

## Final set of rules and exceptions

Based on this analysis, we conclude the rules to 12 rules and 78 exceptions. 90 bits of information to confidently cover +1800 of the most common nouns.

| Rule No. | Noun ending | Gender | Count in dataset | No of exceptions | Exceptions | (in english) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | -o  | masculine | 582 | 2 | mano, radio | hand, radio |
| 2 | -ema | masculine | 6 | 1 | crema | cream |
| 3 | -a | feminine | 595 | 19 | día, planeta, fantasma, programa, clima, mapa, rosa, drama, camarada, mediodía, terrorista, alfa, trauma, vodka, poeta, periodista, psicópata, granuja, pirata | day, planet, ghost, program, weather, map, pink, drama, comrade, noon, terrorist, alpha, trauma, vodka, poet, journalist, psycho, rascal, pirate |
| 4 | -ción or -sión | feminine | 120 | 0 |  |  |
| 5 | -n | masculine | 69 | 8 | razón, imagen, opinión, conexión, religión, sartén, región, hinchazón | reason, picture, opinion, connection, religion, pan, region, swell |
| 5 | -r | masculine | 86 | 2 | mujer, flor | woman, flower |
| 6 | -s | masculine | 20 | 1 | crisis | crisis |
| 7 | -l | masculine | 51 | 4 | señal, cárcel, piel, sal | sign, jail, skin, salt |
| 8 | -d | feminine | 54 | 3 | césped, récord, ataúd | grass, record, coffin |
| 9 | -uz or -iz | feminine | 5 | 0 |  |  |
| 10 | -z | masculine | 11 | 4 | voz, paz, nuez, raíz | voice, peace, nut, root |
| 11 | -e | masculine | 129 | 31 | gente, noche, madre, mente, muerte, parte, sangre, suerte, calle, clase, llave, tarde, fe, leche, carne, base, nieve, madame, medianoche, fuente, superficie, torre, serpiente, serie, corriente, especie, fiebre, nube, fase, hambre, suite | people, night, mother, mind, death, part, blood, luck, street, class, key, afternoon, faith, milk, meat, base, snow, madame, midnight, source, surface, tower, snake, series, current, species, fever, cloud, phase, hunger, suite |
| 12 | every other ending | masculine | 35 | 3 | mamá, ley, tribu | mum, law, tribe |

### Exceptions as a plot

To help remember the exceptions for these rules, we can try and form patterns around certain words i.e., mujer, madre, mamá are all feminine exceptions to each respective rule. 

To achieve this, we extract BERT embeddings of each word, use a UMAP projection to reduce the dimensionality and plot the projections in a Euclidean space. We can then cluster the data to form trivial groups of words. First we identify all the exceptions and add them to a list.

<details>
<summary>Click to expand Python code</summary>

```python
# identify all exceptions and add to list
exceptions_spanish = []
exceptions_english = []
exceptions_gender = []

# fetch words ending in 'o' that are feminine
exceptions_spanish.extend(df_o[(df_o['gender'] == 'feminine')]['spanish'].tolist())
exceptions_english.extend(df_o[(df_o['gender'] == 'feminine')]['english'].tolist())
exceptions_gender.extend(df_o[(df_o['gender'] == 'feminine')]['gender'].tolist())

# fetch words ending in 'ema' that are feminine
exceptions_spanish.extend(df_noboth[(df_noboth['spanish'].str.endswith('ema')) & (df_noboth['gender'] == 'feminine')]['spanish'].tolist())
exceptions_english.extend(df_noboth[(df_noboth['spanish'].str.endswith('ema')) & (df_noboth['gender'] == 'feminine')]['english'].tolist())
exceptions_gender.extend(df_noboth[(df_noboth['spanish'].str.endswith('ema')) & (df_noboth['gender'] == 'feminine')]['gender'].tolist())

# fetch words ending in 'a' that are masculine
exceptions_spanish.extend(df_a[(~df_a['spanish'].str.endswith('ema')) & (df_a['gender'] == 'masculine')]['spanish'].tolist())
exceptions_english.extend(df_a[(~df_a['spanish'].str.endswith('ema')) & (df_a['gender'] == 'masculine')]['english'].tolist())
exceptions_gender.extend(df_a[(~df_a['spanish'].str.endswith('ema')) & (df_a['gender'] == 'masculine')]['gender'].tolist())

# fetch words ending in 'n' that are feminine
exceptions_spanish.extend(df_n[(df_n['gender'] == 'feminine')]['spanish'].tolist())
exceptions_english.extend(df_n[(df_n['gender'] == 'feminine')]['english'].tolist())
exceptions_gender.extend(df_n[(df_n['gender'] == 'feminine')]['gender'].tolist())

# fetch words ending in 'r' that are feminine
exceptions_spanish.extend(df_r[(df_r['gender'] == 'feminine')]['spanish'].tolist())
exceptions_english.extend(df_r[(df_r['gender'] == 'feminine')]['english'].tolist())
exceptions_gender.extend(df_r[(df_r['gender'] == 'feminine')]['gender'].tolist())

# fetch words ending in 's' that are feminine
exceptions_spanish.extend(df_s[(df_s['gender'] == 'feminine')]['spanish'].tolist())
exceptions_english.extend(df_s[(df_s['gender'] == 'feminine')]['english'].tolist())
exceptions_gender.extend(df_s[(df_s['gender'] == 'feminine')]['gender'].tolist())

# fetch words ending in 'l' that are feminine
exceptions_spanish.extend(df_l[(df_l['gender'] == 'feminine')]['spanish'].tolist())
exceptions_english.extend(df_l[(df_l['gender'] == 'feminine')]['english'].tolist())
exceptions_gender.extend(df_l[(df_l['gender'] == 'feminine')]['gender'].tolist())

# fetch words ending in 'd' that are feminine
exceptions_spanish.extend(df_d[(df_d['gender'] == 'masculine')]['spanish'].tolist())
exceptions_english.extend(df_d[(df_d['gender'] == 'masculine')]['english'].tolist())
exceptions_gender.extend(df_d[(df_d['gender'] == 'masculine')]['gender'].tolist())

# fetch words ending in 'z' but no -uz or -iz that are feminine
exceptions_spanish.extend(df_reduced[(df_reduced['spanish'].str.endswith('z')) & (~df_reduced['spanish'].str.endswith('uz')) &\
                                     (~df_reduced['spanish'].str.endswith('iz')) & (df_reduced['gender'] == 'feminine')]['spanish'].tolist())
exceptions_english.extend(df_reduced[(df_reduced['spanish'].str.endswith('z')) & (~df_reduced['spanish'].str.endswith('uz')) &\
                                     (~df_reduced['spanish'].str.endswith('iz')) & (df_reduced['gender'] == 'feminine')]['english'].tolist())
exceptions_gender.extend(df_reduced[(df_reduced['spanish'].str.endswith('z')) & (~df_reduced['spanish'].str.endswith('uz')) &\
                                     (~df_reduced['spanish'].str.endswith('iz')) & (df_reduced['gender'] == 'feminine')]['gender'].tolist())

# fetch words not ending 'e' or 'z' that are feminine
exceptions_spanish.extend(df_reduced[(~df_reduced['spanish'].str.endswith('e')) & (~df_reduced['spanish'].str.endswith('z')) &\
                                     (df_reduced['gender'] == 'feminine')]['spanish'].tolist())
exceptions_english.extend(df_reduced[(~df_reduced['spanish'].str.endswith('e')) & (~df_reduced['spanish'].str.endswith('z')) &\
                                     (df_reduced['gender'] == 'feminine')]['english'].tolist())
exceptions_gender.extend(df_reduced[(~df_reduced['spanish'].str.endswith('e')) & (~df_reduced['spanish'].str.endswith('z')) &\
                                     (df_reduced['gender'] == 'feminine')]['gender'].tolist())

# fetch words ending 'e' that are feminine
exceptions_spanish.extend(df_reduced[(df_reduced['spanish'].str.endswith('e')) & (df_reduced['gender'] == 'feminine')]['spanish'].tolist())
exceptions_english.extend(df_reduced[(df_reduced['spanish'].str.endswith('e')) & (df_reduced['gender'] == 'feminine')]['english'].tolist())
exceptions_gender.extend(df_reduced[(df_reduced['spanish'].str.endswith('e')) & (df_reduced['gender'] == 'feminine')]['gender'].tolist())
```

</details>

Then we use BERT to extract features.

<details>
<summary>Click to expand Python code</summary>

```python
# %pip install transformers
from transformers import BertTokenizer, BertModel
from transformers import pipeline
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

# encode the text using bert
def bert_encode(x):
  encoded_input = tokenizer(x, return_tensors='pt')
  output = model(**encoded_input)
  return pd.DataFrame(output['pooler_output'].detach().numpy()).T

# get embeddings for all words in exceptions_english
series_list = []
for word in exceptions_english:
    series_list.append(pd.Series(bert_encode([str(word)])[0]))

X_bert = pd.DataFrame(series_list)
```

</details>

Lastly, visualise UMAP projections of the BERT embeddings with coloured KMeans clusters with Spanish nouns and their english translations.

<details>
<summary>Click to expand Python code</summary>

```python
# visualise words in 2D space based on embeddings
from umap.umap_ import UMAP
from sklearn.cluster import KMeans
import seaborn as sns

umap = UMAP(n_components=2, random_state=42, min_dist=0.85)
umap_embedding = umap.fit_transform(X_bert)

# get cluster labels
kmeans = KMeans(n_clusters=7, random_state=42).fit(X_bert)

# create a dataframe from the embeddings
umap_df = pd.DataFrame(umap_embedding, columns=['Dimension 1', 'Dimension 2'])
umap_df['cluster'] = kmeans.labels_
umap_df['gender'] = exceptions_gender

# plot the embeddings
plt.figure(figsize=(14, 6))
sns.scatterplot(x='Dimension 1', y='Dimension 2', data=umap_df, alpha=0.5, s=100, hue='cluster', palette='tab10', style='gender')

# add annotations one by one with a loop
for i in range(len(umap_df)):
    plt.text(umap_df['Dimension 1'][i], umap_df['Dimension 2'][i], exceptions_spanish[i], fontsize=10)

# remove legend for hue, but keep legend for style
plt.gca().get_legend().remove()

plt.title('UMAP projection of BERT embeddings for Spanish nouns\n(points marked with x are masculine, o are feminine)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
```

</details>

<p align="center">
  <img src="/assets/blog-1/Untitled%2025.png" width="1000" />
</p>

<p align="center">
  <img src="/assets/blog-1/Untitled%2026.png" width="1000" />
</p>

## Future directions

- If I were to continue this analysis, my next step would be to consider adding in etymological features. If anyone knows who to get this through an API, please let me know.
- Considering there are only 78 exceptions, a follow up to this might involve an explanation as to why these nouns don’t follow the rules.
- The original goal of identifying the gender of nouns was to know when to use el/la articles. However I later found out that there are some cases where the gender doesn’t matter. This particularly is the case for feminine nouns that have stressed "a" or “ha” in its first syllable, they actually use el instead. More info [here](https://spanish.kwiziq.com/revision/grammar/feminine-nouns-starting-with-a-stressed-a-take-masculine-articles-and-quantifiers). We provide a list of these instances below:
    - el agua (water)
    - el alma (soul)
    - el águila (eagle)
    - el arma (weapon)
    - el ala (wing)
    - el asta (flagpole)
    - el arca (ark)
    - el arpa (harp)
    - el asma (asthma)
    - el álgebra (algebra)
    - el hambre (hunger)
    - el hacha (axe)