# Token Classification for Cyber security dataset 
---

We are planning to use the NLP transformers available in Hugging face repo to perform token clasification

## Install Libraries
---


```python
! pip install datasets transformers seqeval
! apt install git-lfs
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: datasets in /usr/local/lib/python3.7/dist-packages (2.7.0)
    Requirement already satisfied: transformers in /usr/local/lib/python3.7/dist-packages (4.24.0)
    Requirement already satisfied: seqeval in /usr/local/lib/python3.7/dist-packages (1.2.2)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.7/dist-packages (from datasets) (0.70.14)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.7/dist-packages (from datasets) (3.8.3)
    Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from datasets) (21.3)
    Requirement already satisfied: dill<0.3.7 in /usr/local/lib/python3.7/dist-packages (from datasets) (0.3.6)
    Requirement already satisfied: huggingface-hub<1.0.0,>=0.2.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (0.11.0)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from datasets) (1.21.6)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.7/dist-packages (from datasets) (3.1.0)
    Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.7/dist-packages (from datasets) (2022.10.0)
    Requirement already satisfied: responses<0.19 in /usr/local/lib/python3.7/dist-packages (from datasets) (0.18.0)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from datasets) (4.13.0)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from datasets) (1.3.5)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.7/dist-packages (from datasets) (6.0)
    Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.7/dist-packages (from datasets) (4.64.1)
    Requirement already satisfied: pyarrow>=6.0.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (6.0.1)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (2.23.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (6.0.2)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.3.3)
    Requirement already satisfied: asynctest==0.13.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (0.13.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.8.1)
    Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (2.1.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (4.0.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (22.1.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: typing-extensions>=3.7.4 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (4.1.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0.0,>=0.2.0->datasets) (3.8.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->datasets) (3.0.9)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (1.25.11)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (2022.9.24)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (2.10)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (2022.6.2)
    Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /usr/local/lib/python3.7/dist-packages (from transformers) (0.13.2)
    Requirement already satisfied: scikit-learn>=0.21.3 in /usr/local/lib/python3.7/dist-packages (from seqeval) (1.0.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.21.3->seqeval) (3.1.0)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.21.3->seqeval) (1.2.0)
    Requirement already satisfied: scipy>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.21.3->seqeval) (1.7.3)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->datasets) (3.10.0)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->datasets) (2022.6)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas->datasets) (1.15.0)
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    git-lfs is already the newest version (2.3.4-1).
    The following package was automatically installed and is no longer required:
      libnvidia-common-460
    Use 'apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 5 not upgraded.


### Huggin Face Login
---


```python
##Hugging face login
from huggingface_hub import notebook_login

notebook_login()
```

    Token is valid.
    Your token has been saved to /root/.huggingface/token
    Login successful


### Transformer Import 
---


```python
import transformers

print(transformers.__version__)
```

    4.24.0


### Token Classification


```python
task = "ner" 
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
from datasets import load_dataset, load_metric
```

### Import Tokenizer
---


```python
from transformers import AutoTokenizer
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
import transformers
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
```

## Import Actual Data from txt files
---
Upload the files to the colab directory and run the program 
Note : The uploaded files will be refreshed for every instance of colab. So we need to upload them everytime a new instance is created



```python
import pandas as pd
def loadRawDatafromTXT(file,delimiterVal):
  dataFrame = pd.read_csv(file,delimiter=delimiterVal,on_bad_lines="skip",header=None,usecols=[0,1])
  ## now clean the data
  dataFrame = dataFrame.dropna()

  ## Perform Further Cleaning for irregular rows
  numRows = dataFrame.shape[0]

  minlength = 3
  maxlength = 20

  def lengthChecker(s):
    s = str(s)
    return len(s) < minlength or len(s) > maxlength
  

  def removeBadRows(dd,col,value):
     dd = dd[dd[col] != value]
     dd = dd.iloc[:,0:2].copy()
     return dd

  def assignColumns(dd,colName):
    dd.columns = colName
    return dd

  additionalColumn = []
  for i in range(numRows):
    individualRow = dataFrame.iloc[i,:][0]
    if(lengthChecker(individualRow)):
      additionalColumn.append("Bad Value")
    else:
      additionalColumn.append("Good Value")
    
  for i in range(numRows):
    individualRow = dataFrame.iloc[i,:][1]
    if(len(str(individualRow)) == 1):
      dataFrame.iloc[i,:][1] = '0'
  
  dataFrame["ValueTested"] = additionalColumn


  dataFrame = removeBadRows(dataFrame,"ValueTested","Bad Value")
  dataFrame = assignColumns(dataFrame,["tokens",'ner_tags'])


  return dataFrame
```


```python
from datasets import load_dataset
import pandas as pd

dataFrame_Train = loadRawDatafromTXT("train.txt","\t")
dataFrame_Test = loadRawDatafromTXT("test.txt","\t")
dataFrame_Validation = loadRawDatafromTXT("valid.txt","\t")


```

### save these Dataframes into pickle oject 




```python
import pickle

with open("TrainDF","wb") as f:
  pickle.dump(dataFrame_Train,f)

with open("TestDF","wb") as f:
  pickle.dump(dataFrame_Test,f)

with open("ValidDF","wb") as f:
  pickle.dump(dataFrame_Validation,f)
```

### Load Pickle  objects into Dataframes
---


```python
with open("TrainDF","rb") as f:
  Train_DataDF = pickle.load(f)

with open("TestDF","rb") as f:
  Test_DataDF = pickle.load(f)

with open("ValidDF","rb") as f:
  Valid_DataDF = pickle.load(f)
```

## Encoding of prediction Tags
---


```python
def LabelGeneration(dfList):
  newLabel = []
  for i in range(3):
    newLabel.extend(list(dfList[i]['ner_tags']))
  
  le = preprocessing.LabelEncoder()
  le.fit(newLabel)


  for i in range(3):
    dfList[i]['ner_tags'] = le.transform(dfList[i]['ner_tags'])
  
  
  labelList = list(set(newLabel))

  def modifyArray(df,k):
    row = -1
    col1 = list(df["tokens"].to_numpy().reshape(row,k))
    col2 = list(df["ner_tags"].to_numpy().reshape(row,k))

    newZip = zip(col1,col2)
    newZipList = list(newZip)

    return newZipList


  def computeDropVal(df,k):
    return df.shape[0] % k;
  


  
  for i in range(3):
    dv = computeDropVal(dfList[i],6)
    if(dv != 0): 
      dfList[i] = dfList[i].iloc[:-dv,:]
    nzl = modifyArray(dfList[i],6)
    newdf = pd.DataFrame(nzl,columns=['tokens','ner_tags'])
    dfList[i] = newdf.copy(deep=True)
  

  return labelList,dfList[0],dfList[1],dfList[2]

```


```python
label_list, Train_DataDF_new,Test_DataDF_new,Valid_DataDF_new =  LabelGeneration([Train_DataDF,Test_DataDF,Valid_DataDF])

```

### Check the data


```python
Valid_DataDF_new[:10]
```





  <div id="df-7ede606e-066b-4b1d-9e3f-ec39fc2476b1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tokens</th>
      <th>ner_tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Riltok, mobile, Trojan, banker, with, global]</td>
      <td>[2, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[reach, JUN, 2019, Riltok, one, numerous]</td>
      <td>[0, 0, 0, 2, 0, 0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[families, mobile, banking, Trojans, with, sta...</td>
      <td>[0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[for, such, malware, functions, and, distribut...</td>
      <td>[0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[methods, Originally, intended, target, the, R...</td>
      <td>[0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[audience, the, banker, was, later, adapted]</td>
      <td>[0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[with, minimal, modifications, for, the, Europ...</td>
      <td>[0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[market., The, bulk, its, victims, more]</td>
      <td>[0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[than, reside, Russia, with, France, second]</td>
      <td>[0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[place, Third, place, shared, Italy, Ukraine]</td>
      <td>[0, 0, 0, 0, 0, 0]</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7ede606e-066b-4b1d-9e3f-ec39fc2476b1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7ede606e-066b-4b1d-9e3f-ec39fc2476b1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7ede606e-066b-4b1d-9e3f-ec39fc2476b1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# Load the data into Dataset Class
---


```python
from datasets import Dataset

def loadDataset(df,split):
  return Dataset.from_pandas(df,split=split)

trainning = loadDataset(Train_DataDF_new,"train")
testing = loadDataset(Test_DataDF_new,"test")
validation = loadDataset(Valid_DataDF_new,"validation")

```

## Tokenisation of inputs
---


```python
label_all_tokens = True
```


```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```

Check tokenisation of the given data


```python
tokenize_and_align_labels(trainning[:10])
```




    {'input_ids': [[101, 3565, 7986, 2448, 15451, 8059, 2852, 9314, 17364, 9350, 102], [101, 27911, 2015, 2293, 7986, 1998, 20421, 2021, 102], [101, 15451, 8059, 6048, 2261, 2420, 2067, 2626, 102], [101, 2055, 11924, 28791, 2099, 23445, 8349, 20540, 102], [101, 1996, 3565, 7986, 2448, 2208, 2005, 102], [101, 11924, 2031, 2179, 2178, 6013, 15451, 8059, 102], [101, 20540, 1996, 3565, 7986, 2448, 11924, 102], [101, 10439, 1998, 2023, 2051, 2038, 2579, 102], [101, 1996, 2433, 2852, 9314, 17364, 9350, 6556, 3229, 102], [101, 23445, 6947, 8400, 2626, 2055, 1996, 2852, 9314, 17364, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'labels': [[-100, 2, 7, 7, 7, 7, 2, 2, 2, 7, -100], [-100, 0, 0, 0, 4, 0, 4, 0, -100], [-100, 0, 0, 0, 0, 0, 0, 0, -100], [-100, 0, 4, 2, 2, 0, 0, 0, -100], [-100, 0, 4, 9, 9, 0, 0, -100], [-100, 4, 0, 0, 0, 0, 0, 0, -100], [-100, 0, 0, 4, 9, 9, 4, -100], [-100, 0, 0, 0, 0, 0, 0, -100], [-100, 0, 0, 2, 2, 2, 7, 0, 0, -100], [-100, 0, 3, 3, 0, 0, 0, 2, 2, 2, -100]]}



To apply this function on all the sentences (or pairs of sentences) in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command.

### Tokenise the entire Data
---


```python
def Tokensization(data,func,b):
  return data.map(func,batched=b)

TokenTrain = Tokensization(trainning,tokenize_and_align_labels,True)
TokenTest = Tokensization(testing,tokenize_and_align_labels,True)
TokenValid = Tokensization(validation,tokenize_and_align_labels,True)

```


      0%|          | 0/4 [00:00<?, ?ba/s]



      0%|          | 0/1 [00:00<?, ?ba/s]



      0%|          | 0/3 [00:00<?, ?ba/s]



```python
TokenTrain[:15]
```




    {'tokens': [['Super', 'Mario', 'Run', 'Malware', 'DroidJack', 'RAT'],
      ['Gamers', 'love', 'Mario', 'and', 'Pokemon', 'but'],
      ['malware', 'authors', 'few', 'days', 'back', 'wrote'],
      ['about', 'Android', 'Marcher', 'trojan', 'variant', 'posing'],
      ['the', 'Super', 'Mario', 'Run', 'game', 'for'],
      ['Android', 'have', 'found', 'another', 'instance', 'malware'],
      ['posing', 'the', 'Super', 'Mario', 'Run', 'Android'],
      ['app', 'and', 'this', 'time', 'has', 'taken'],
      ['the', 'form', 'DroidJack', 'RAT', 'remote', 'access'],
      ['trojan', 'Proofpoint', 'wrote', 'about', 'the', 'DroidJack'],
      ['RAT', 'side-loaded', 'with', 'the', 'Pokemon', 'app'],
      ['back', 'July', '2016', 'the', 'difference', 'here'],
      ['that', 'there', 'game', 'included', 'the', 'malicious'],
      ['package', 'The', 'authors', 'are', 'trying', 'latch'],
      ['onto', 'the', 'popularity', 'the', 'Super', 'Mario']],
     'ner_tags': [[2, 7, 7, 7, 2, 7],
      [0, 0, 4, 0, 4, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 4, 2, 0, 0, 0],
      [0, 4, 9, 9, 0, 0],
      [4, 0, 0, 0, 0, 0],
      [0, 0, 4, 9, 9, 4],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 2, 7, 0, 0],
      [0, 3, 0, 0, 0, 2],
      [7, 0, 0, 0, 4, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 4, 9]],
     'input_ids': [[101,
       3565,
       7986,
       2448,
       15451,
       8059,
       2852,
       9314,
       17364,
       9350,
       102],
      [101, 27911, 2015, 2293, 7986, 1998, 20421, 2021, 102],
      [101, 15451, 8059, 6048, 2261, 2420, 2067, 2626, 102],
      [101, 2055, 11924, 28791, 2099, 23445, 8349, 20540, 102],
      [101, 1996, 3565, 7986, 2448, 2208, 2005, 102],
      [101, 11924, 2031, 2179, 2178, 6013, 15451, 8059, 102],
      [101, 20540, 1996, 3565, 7986, 2448, 11924, 102],
      [101, 10439, 1998, 2023, 2051, 2038, 2579, 102],
      [101, 1996, 2433, 2852, 9314, 17364, 9350, 6556, 3229, 102],
      [101, 23445, 6947, 8400, 2626, 2055, 1996, 2852, 9314, 17364, 102],
      [101, 9350, 2217, 1011, 8209, 2007, 1996, 20421, 10439, 102],
      [101, 2067, 2251, 2355, 1996, 4489, 2182, 102],
      [101, 2008, 2045, 2208, 2443, 1996, 24391, 102],
      [101, 7427, 1996, 6048, 2024, 2667, 25635, 102],
      [101, 3031, 1996, 6217, 1996, 3565, 7986, 102]],
     'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1]],
     'labels': [[-100, 2, 7, 7, 7, 7, 2, 2, 2, 7, -100],
      [-100, 0, 0, 0, 4, 0, 4, 0, -100],
      [-100, 0, 0, 0, 0, 0, 0, 0, -100],
      [-100, 0, 4, 2, 2, 0, 0, 0, -100],
      [-100, 0, 4, 9, 9, 0, 0, -100],
      [-100, 4, 0, 0, 0, 0, 0, 0, -100],
      [-100, 0, 0, 4, 9, 9, 4, -100],
      [-100, 0, 0, 0, 0, 0, 0, -100],
      [-100, 0, 0, 2, 2, 2, 7, 0, 0, -100],
      [-100, 0, 3, 3, 0, 0, 0, 2, 2, 2, -100],
      [-100, 7, 0, 0, 0, 0, 0, 4, 0, -100],
      [-100, 0, 0, 0, 0, 0, 0, -100],
      [-100, 0, 0, 0, 0, 0, 0, -100],
      [-100, 0, 0, 0, 0, 0, 0, -100],
      [-100, 0, 0, 0, 0, 4, 9, -100]]}



## TRAIN THE MODEL
---


```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
```


    Downloading:   0%|          | 0.00/268M [00:00<?, ?B/s]


    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForTokenClassification: ['vocab_projector.bias', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_transform.weight', 'vocab_layer_norm.weight']
    - This IS expected if you are initializing DistilBertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of DistilBertForTokenClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=4,
    weight_decay=0.03,
    push_to_hub=True,
)
```

### Data Collator
---


```python
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)
```

## Metrics
---


```python
metric = load_metric("seqeval")
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
      """Entry point for launching an IPython kernel.



    Downloading builder script:   0%|          | 0.00/2.47k [00:00<?, ?B/s]


## Metrics Evaluation 
---


```python
import numpy as np

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```

Note that we drop the precision/recall/f1 computed for each category and only focus on the overall precision/recall/f1/accuracy.

Then we just need to pass all of this along with our datasets to the `Trainer`:

# Model Trainning
---


```python
trainer = Trainer(
    model,
    args,
    train_dataset=TokenTrain,
    eval_dataset=TokenValid,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

    Cloning https://huggingface.co/Thivin/distilbert-base-uncased-finetuned-ner into local empty directory.
    WARNING:huggingface_hub.repository:Cloning https://huggingface.co/Thivin/distilbert-base-uncased-finetuned-ner into local empty directory.



    Download file pytorch_model.bin:   0%|          | 1.94k/253M [00:00<?, ?B/s]



    Download file runs/Nov18_09-40-43_e3d4f73c473e/events.out.tfevents.1668764499.e3d4f73c473e.3416.2:  46%|####6 …



    Download file runs/Nov18_09-08-03_e3d4f73c473e/events.out.tfevents.1668762643.e3d4f73c473e.3416.0:  87%|######…



    Download file runs/Nov18_09-40-43_e3d4f73c473e/1668764499.4639833/events.out.tfevents.1668764499.e3d4f73c473e.…



    Download file training_args.bin: 100%|##########| 3.36k/3.36k [00:00<?, ?B/s]



    Download file runs/Nov18_10-05-41_e3d4f73c473e/1668765968.6934247/events.out.tfevents.1668765968.e3d4f73c473e.…



    Download file runs/Nov18_10-05-41_e3d4f73c473e/events.out.tfevents.1668765968.e3d4f73c473e.3416.4:  59%|#####9…



    Download file runs/Nov18_09-08-03_e3d4f73c473e/1668762643.443948/events.out.tfevents.1668762643.e3d4f73c473e.3…



    Clean file runs/Nov18_09-40-43_e3d4f73c473e/events.out.tfevents.1668764499.e3d4f73c473e.3416.2:  23%|##3      …



    Clean file runs/Nov18_09-08-03_e3d4f73c473e/events.out.tfevents.1668762643.e3d4f73c473e.3416.0:  25%|##4      …



    Clean file runs/Nov18_09-40-43_e3d4f73c473e/1668764499.4639833/events.out.tfevents.1668764499.e3d4f73c473e.341…



    Clean file training_args.bin:  30%|##9       | 1.00k/3.36k [00:00<?, ?B/s]



    Clean file runs/Nov18_10-05-41_e3d4f73c473e/1668765968.6934247/events.out.tfevents.1668765968.e3d4f73c473e.341…



    Clean file runs/Nov18_10-05-41_e3d4f73c473e/events.out.tfevents.1668765968.e3d4f73c473e.3416.4:  17%|#6       …



    Clean file runs/Nov18_09-08-03_e3d4f73c473e/1668762643.443948/events.out.tfevents.1668762643.e3d4f73c473e.3416…



    Download file runs/Nov18_10-05-41_e3d4f73c473e/events.out.tfevents.1668768228.e3d4f73c473e.3416.6: 100%|######…



    Clean file runs/Nov18_10-05-41_e3d4f73c473e/events.out.tfevents.1668768228.e3d4f73c473e.3416.6: 100%|#########…



    Clean file pytorch_model.bin:   0%|          | 1.00k/253M [00:00<?, ?B/s]


We can now finetune our model by just calling the `train` method:


```python
trainer.train()
```

    The following columns in the training set don't have a corresponding argument in `DistilBertForTokenClassification.forward` and have been ignored: tokens, ner_tags. If tokens, ner_tags are not expected by `DistilBertForTokenClassification.forward`,  you can safely ignore this message.
    /usr/local/lib/python3.7/dist-packages/transformers/optimization.py:310: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      FutureWarning,
    ***** Running training *****
      Num examples = 3136
      Num Epochs = 4
      Instantaneous batch size per device = 16
      Total train batch size (w. parallel, distributed & accumulation) = 16
      Gradient Accumulation steps = 1
      Total optimization steps = 784
      Number of trainable parameters = 66371339
    You're using a DistilBertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.




    <div>

      <progress value='784' max='784' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [784/784 30:52, Epoch 4/4]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.299171</td>
      <td>0.769847</td>
      <td>0.598082</td>
      <td>0.673181</td>
      <td>0.912727</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>0.384625</td>
      <td>0.795125</td>
      <td>0.625551</td>
      <td>0.700218</td>
      <td>0.921877</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.204500</td>
      <td>0.368258</td>
      <td>0.775173</td>
      <td>0.668308</td>
      <td>0.717785</td>
      <td>0.925455</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.204500</td>
      <td>0.408273</td>
      <td>0.770448</td>
      <td>0.668826</td>
      <td>0.716049</td>
      <td>0.924457</td>
    </tr>
  </tbody>
</table><p>


    The following columns in the evaluation set don't have a corresponding argument in `DistilBertForTokenClassification.forward` and have been ignored: tokens, ner_tags. If tokens, ner_tags are not expected by `DistilBertForTokenClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 2282
      Batch size = 16
    /usr/local/lib/python3.7/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: 0 seems not to be NE tag.
      warnings.warn('{} seems not to be NE tag.'.format(chunk))
    /usr/local/lib/python3.7/dist-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    The following columns in the evaluation set don't have a corresponding argument in `DistilBertForTokenClassification.forward` and have been ignored: tokens, ner_tags. If tokens, ner_tags are not expected by `DistilBertForTokenClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 2282
      Batch size = 16
    Saving model checkpoint to distilbert-base-uncased-finetuned-ner/checkpoint-500
    Configuration saved in distilbert-base-uncased-finetuned-ner/checkpoint-500/config.json
    Model weights saved in distilbert-base-uncased-finetuned-ner/checkpoint-500/pytorch_model.bin
    tokenizer config file saved in distilbert-base-uncased-finetuned-ner/checkpoint-500/tokenizer_config.json
    Special tokens file saved in distilbert-base-uncased-finetuned-ner/checkpoint-500/special_tokens_map.json
    tokenizer config file saved in distilbert-base-uncased-finetuned-ner/tokenizer_config.json
    Special tokens file saved in distilbert-base-uncased-finetuned-ner/special_tokens_map.json
    The following columns in the evaluation set don't have a corresponding argument in `DistilBertForTokenClassification.forward` and have been ignored: tokens, ner_tags. If tokens, ner_tags are not expected by `DistilBertForTokenClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 2282
      Batch size = 16
    /usr/local/lib/python3.7/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: 0 seems not to be NE tag.
      warnings.warn('{} seems not to be NE tag.'.format(chunk))
    /usr/local/lib/python3.7/dist-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    The following columns in the evaluation set don't have a corresponding argument in `DistilBertForTokenClassification.forward` and have been ignored: tokens, ner_tags. If tokens, ner_tags are not expected by `DistilBertForTokenClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 2282
      Batch size = 16
    
    
    Training completed. Do not forget to share your model on huggingface.co/models =)
    
    





    TrainOutput(global_step=784, training_loss=0.1502026088383733, metrics={'train_runtime': 1856.7267, 'train_samples_per_second': 6.756, 'train_steps_per_second': 0.422, 'total_flos': 59215903557792.0, 'train_loss': 0.1502026088383733, 'epoch': 4.0})



## Evaluate Model 
---


```python
trainer.evaluate()
```

    The following columns in the evaluation set don't have a corresponding argument in `DistilBertForTokenClassification.forward` and have been ignored: tokens, ner_tags. If tokens, ner_tags are not expected by `DistilBertForTokenClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 2282
      Batch size = 16




<div>

  <progress value='143' max='143' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [143/143 00:55]
</div>






    {'eval_loss': 0.40827253460884094,
     'eval_precision': 0.7704477611940298,
     'eval_recall': 0.6688261207566727,
     'eval_f1': 0.7160493827160493,
     'eval_accuracy': 0.9244574780058651,
     'eval_runtime': 56.5125,
     'eval_samples_per_second': 40.38,
     'eval_steps_per_second': 2.53,
     'epoch': 4.0}



## Generate Predicitons
---



```python
predictions, labels, _ = trainer.predict(TokenValid)
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results
```

    The following columns in the test set don't have a corresponding argument in `DistilBertForTokenClassification.forward` and have been ignored: tokens, ner_tags. If tokens, ner_tags are not expected by `DistilBertForTokenClassification.forward`,  you can safely ignore this message.
    ***** Running Prediction *****
      Num examples = 2282
      Batch size = 16






    /usr/local/lib/python3.7/dist-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: 0 seems not to be NE tag.
      warnings.warn('{} seems not to be NE tag.'.format(chunk))
    /usr/local/lib/python3.7/dist-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))





    {'Indicator': {'precision': 0.6013745704467354,
      'recall': 0.6704980842911877,
      'f1': 0.6340579710144928,
      'number': 261},
     'Malware': {'precision': 0.6301369863013698,
      'recall': 0.4876325088339223,
      'f1': 0.549800796812749,
      'number': 283},
     'Organization': {'precision': 0.44,
      'recall': 0.09217877094972067,
      'f1': 0.15242494226327946,
      'number': 358},
     'System': {'precision': 0.5486725663716814,
      'recall': 0.4381625441696113,
      'f1': 0.48722986247544203,
      'number': 283},
     'Vulnerability': {'precision': 0.8314296967309964,
      'recall': 0.7915260592425947,
      'f1': 0.8109873223203995,
      'number': 2667},
     '_': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'number': 7},
     'overall_precision': 0.7704477611940298,
     'overall_recall': 0.6688261207566727,
     'overall_f1': 0.7160493827160493,
     'overall_accuracy': 0.9244574780058651}




```python

```
