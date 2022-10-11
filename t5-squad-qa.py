import json
from tracemalloc import start
from transformers import {
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
}
import torch
import tqdm
from torch.utils.data import DataLoader


def readData(path):
    with open(path, 'rb') as f:
        dict = json.load(f)

    # Finding all the contexts, questions, and answers in the SQUAD dataset
    cntxts = []
    qstns = []
    answrs = []
    # Preprocessing through iteration of the dataset
    for doc in dict['data']:
        for paragraph in doc['paragraphs']:
            cntxt = paragraph['context']
            for qa in paragraph['qas']:
                qstn = qa['question']
                if 'plausible_answers' in qa.keys():
                    toPick = 'plausible_answers'
                else:
                    toPick = 'answers'
                for answr in qa[toPick]:
                    # Appending data to their corresponding lists
                    cntxts.append(cntxt)
                    qstns.append(qstn)
                    answrs.append(answr)
    
    return cntxts, qstns, answrs 

# Grabbing the corresponding data in our desired format for training and validation datasets (validation not formatted yet)
# TODO: Add validation data here
trainingContexts, trainingQuestions, trainingAnswers = readData('./data/train-v2.0.json')

def addEndIndex(answers, contexts):
    # Zip answers and context into answer-context pairs and loop through to process indeces
    for answer, context in zip(answers, contexts):
        desiredText = answer['text']
        startIndex = answer['answer_start']
        endIndex = startIndex + len(desiredText)

        # Squad dataset can be off by one or two characters to the left, so we account for this
        if context[startIndex:endIndex] != desiredText:
            ctr = 1
            while ctr < 3:
                if context[startIndex-ctr:endIndex-ctr] == desiredText:
                    answer['answer_start'] = startIndex - ctr
                    answer['answer_end'] = endIndex - ctr
                    break
                ctr += 1
        else:
            answer['answer_end'] = endIndex

addEndIndex(trainingAnswers, trainingContexts)

MODEL_NAME = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Getting the encodings from our tokenizer for both our training and validation dataset
# TODO: Add the validation dataset here
sourceEncodings = tokenizer(
                    trainingQuestions, 
                    trainingContexts,  
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                  )

def getOnlyText(answers):
    trainingAnswerTexts = []
    for answer in answers:
        trainingAnswerTexts.append(answer['text'])
    return trainingAnswerTexts


answerEncodings = tokenizer(
                    getOnlyText(trainingAnswers),
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                  )

def addTokenPositions(encodings, answers):
    startPositions = []
    endPositions = []
    for i in range(len(answers)):
        startPositions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        endPositions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # If no start position is None for our current index (i.e. the last index), then it got truncated
        if startPositions[-1] is None:
            startPositions[-1] = tokenizer.model_max_length

        # If our current end position is None, then we found a space, so we need to go back until next valid token
        validShift = 1
        while endPositions[-1] is None:
            endPositions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - validShift)
            validShift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': startPositions, 'end_positions': endPositions})

# Add token positions for our encodings given our answers for both the training and validation datasets
# TODO: Add validation dataset here
addTokenPositions(sourceEncodings, trainingAnswers)


# Recommended class from documentation
class Data(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Creating dictionary class for torch tensor values given the encodings
trainingData = Data(trainingEncodings)

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# move model over to detected device
model.to(device)
# activate training mode of model
model.train()
# initialize adam optimizer with weight decay (reduces chance of overfitting)
adamOptimizer = AdamW(model.parameters(), lr=5e-5)

# initialize data loader for training data
trainingLoader = DataLoader(trainingData, batch_size=16, shuffle=True)

for epoch in range(3):
    # set model to train mode
    model.train()
    # setup loop (we use tqdm for the progress bar)
    loop = tqdm(trainingLoader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        adamOptimizer.zero_grad()
        # pull all the tensor batches required for training
        inputIDs = batch['input_ids'].to(device)
        attentionMask = batch['attention_mask'].to(device)
        startPositions = batch['start_positions'].to(device)
        endPositions = batch['end_positions'].to(device)
        # train model on batch and return outputs (incl. loss)
        outputs = model(inputIDs, attention_mask=attentionMask,
                        start_positions=startPositions,
                        end_positions=endPositions)
        # extract loss
        loss = outputs[0]
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        adamOptimizer.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

model_path = 'model/'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

