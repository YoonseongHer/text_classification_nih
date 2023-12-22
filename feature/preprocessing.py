import re
from sklearn.preprocessing import LabelEncoder

def text_cleansing(text):
    p = re.compile('&[^ ㄱ-힣0-9]*|class=[^ ]*|[◈▲□○◯◦꠲ꡔꡕ]|scope=[^ ]*|'
                   +'colspan=[^ ]*|rowspan=[^ ]*|style=[^ ]*|href=[^ ]*|'
                   +'target=[^ ]*|title=[^ ]*|[｢｣]')
    pred_text = re.sub(' +|\n',' ',re.sub(p,'',text)).strip()
    pred_text = re.sub('[^a-zA-Z0-9ㄱ-힣.,()!? -+=*%@#]','',pred_text)
    return pred_text

def label_encoding(labels):
    le = LabelEncoder()
    encoding_labels = le.fit_transform(labels)
    return encoding_labels, le.classes_

def data_preprocessing(df, le=[]):
    processed_df = df.copy()
    processed_df['data'] = processed_df['data'].apply(text_cleansing)
    if le:
        processed_df['label'] = [le.index(label) for label in processed_df['label']]
        label_class = le
    else:
        processed_df['label'], label_class = label_encoding(processed_df['label'])
    return processed_df, label_class