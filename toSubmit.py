import json
import os
import pickle
import re
from copy import deepcopy
from functools import partial

import fasttext
import pandas as pd
import requests

# Commented out imports (for reference)
#import concurrent.futures
#from concurrent.futures import ThreadPoolExecutor, as_completed




base_url = "http://nodes.flit.to:30091"  # API의 기본 URL


model = fasttext.load_model('lid.176.bin')


def iso_639_1_to_fullname(code):
    iso_639_1 = {
        "ar": "Arabic",
        "cs": "Czech",
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "hi": "Hindi",
        "hu": "Hungarian",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "mn": "Mongolian",
        "ms": "Malay",
        "nl": "Dutch",
        "no": "Norwegian",
        "nb": "Norwegian Bokmål",
        "nn": "Norwegian Nynorsk",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sv": "Swedish",
        "th": "Thai",
        "tl": "Tagalog",
        "tr": "Turkish",
        "uz": "Uzbek",
        "vi": "Vietnamese",
        "zh-CN": "Simplified Chinese",
        "zh-TW": "Traditional Chinese",
    }
    return iso_639_1.get(code, "Unknown")

def fullname_to_iso_639_1(name):
    fullname_to_iso = {
        "Arabic": "ar",
        "Czech": "cs",
        "Danish": "da",
        "German": "de",
        "Greek": "el",
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "Hindi": "hi",
        "Hungarian": "hu",
        "Indonesian": "id",
        "Italian": "it",
        "Japanese": "ja",
        "Korean": "ko",
        "Mongolian": "mn",
        "Malay": "ms",
        "Dutch": "nl",
        "Norwegian": "no",
        "Norwegian Bokmål": "nb",
        "Norwegian Nynorsk": "nn",
        "Polish": "pl",
        "Portuguese": "pt",
        "Romanian": "ro",
        "Russian": "ru",
        "Swedish": "sv",
        "Thai": "th",
        "Tagalog": "tl",
        "Turkish": "tr",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Simplified Chinese": "zh-CN",
        "Traditional Chinese": "zh-TW",
    }
    return fullname_to_iso.get(name, "Unknown")



def flid_predict_language(text, n=1):
    url = f"{base_url}/api/langid/predict"
    payload = {
        "text": text,
        "n": n
    }
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"


# def parallel_language_prediction(df, chunk_size=30, max_workers=10):
#     def predict_language_safe(text):
#         try:
#             result = flid_predict_language(text)
#             if isinstance(result, dict):
#                 return max(result, key=result.get)
#             else:
#                 return 'error'
#         except Exception as e:
#             return f'error: {str(e)}'

#     def process_batch(df_batch):
#         df_batch['predict_src_content_lang'] = df_batch['src_content'].apply(predict_language_safe)
#         df_batch['predict_dst_content_lang'] = df_batch['dst_content'].apply(predict_language_safe)
#         return df_batch

#     df_chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         results = list(executor.map(process_batch, df_chunks))

#     result_df = pd.concat(results)

#     print(result_df.head())

#     with open('result_df.pkl', 'wb') as f:
#         pickle.dump(result_df, f)

#     return result_df


def fast_predict_language(text):
    if not isinstance(text, str):
        return None

    prediction = model.predict([text], k=1)
    language_code = prediction[0][0][0].split('__')[-1]

    return language_code


def contains_special_character(text):
    if not isinstance(text, str):
        return False
    for char in text:
        if char in '"“”,.·*※☆★!?•|:[]()<>/『』「」《+-~%…&°\'':
            return True
    return False



def is_string_instance(text):
    """
    이 함수는 input_text가 문자열(str)의 인스턴스인지 확인합니다.
    
    input_text가 NaN(즉, float)인 경우, isinstance(input_text, str)은 False를 반환합니다.
    
    매개변수:
    input_text (any): 확인할 입력 텍스트.
    
    반환값:
    bool: input_text가 문자열인 경우 True, 그렇지 않은 경우 False.
    """
    if isinstance(text, str):
        return True
    return False


def is_numeric_string(text):
    """
    Checks if the given text is a numeric string.
    
    A numeric string is defined as a string that consists solely of digits,
    and may include a leading negative sign.
    
    Parameters:
    text (str): The text to check.
    
    Returns:
    bool: True if the text is a numeric string, False otherwise.
    """
    if not is_string_instance(text):
        return False
    return bool(re.fullmatch(r'-?\d+', text))


def replace_newline_with_space(text):
    if not isinstance(text, str):
        return None
    return text.replace('\n', ' ')

def detect_korean(text):
    """
    Detects if the given text contains Korean characters.

    Korean characters include Hangul syllables and Jamo.

    Parameters:
    text (str): The text to check.

    Returns:
    str: 'ko' if the text contains Korean characters, None otherwise.
    """
    if not isinstance(text, str):
        return None
    
    if re.search(r"[ㄱ-ㅎㅏ-ㅣ가-힣]", text):
        return True
    else:
        return False

def detect_only_korean(text):
    if not isinstance(text, str):
        return None
    # Check if the text contains only Korean characters (Hangul syllables and Jamo)
    if re.match(r'^[ㄱ-ㅎㅏ-ㅣ가-힣\s]+$', text):
        return True 
    return False 
    

def detect_korean_jamo(text):
    """
    Detects if the given text contains Korean Jamo (e.g. ㄱ, ㄴ, ㄷ, ..., ㅏ, ㅑ, ㅓ, ..., ㅣ).

    Parameters:
    text (str): The text to check.

    Returns:
    str: 'ko' if the text contains Korean Jamo, None otherwise.
    """
    if not isinstance(text, str):
        return None

    if re.search(r"[ㄱ-ㅎㅏ-ㅣ]", text):
        return 'jamo'
    else:
        return None


def detect_japanese(text):
    if not isinstance(text, str):
        return None
    if re.search(r"[ぁ-ゔ]+|[ァ-ヴー]+[々〆〤]", text):
        return True
    else:
        return False
    
def detect_chinese(text):
    """
    Detects if the given text contains Chinese characters.

    Chinese characters are within the Unicode range \u4E00 to \u9FA5.

    Parameters:
    text (str): The text to check.

    Returns:
    str: 'zh' if the text contains Chinese characters, None otherwise.
    """

    if not isinstance(text, str):
        return None

    if re.search(r".*[\u4E00-\u9FA5]+.*", text):
        return True
    else:
        return False

def detect_only_chinese(text):
    """
    Detects if the given text contains only Chinese characters.

    Chinese characters are within the Unicode range \u4E00 to \u9FA5.

    Parameters:
    text (str): The text to check.

    Returns:
    bool: True if the text contains only Chinese characters, False otherwise.
    """
    if not isinstance(text, str):
        return False

    return bool(re.match(r'^[\u4E00-\u9FA5]+$', text))
    
def detect_simplified_chinese(text):
    """
    Detects if the given text contains Simplified Chinese characters.

    Simplified Chinese characters are within the Unicode range \u4e00 to \u9fff.

    Parameters:
    text (str): The text to check.

    Returns:
    bool: True if the text contains Simplified Chinese characters, False otherwise.
    """
    if not isinstance(text, str):
        return None
    for char in text:
        if '\u4E00' <= char <= '\u9FFF':
            return True
    return False

def detect_english(text):
    """
    Detects if the given text contains English characters.

    English characters are within the Unicode ranges A-Z and a-z.

    Parameters:
    text (str): The text to check.

    Returns:
    bool: True if the text contains English characters, False otherwise.
    """
    if not isinstance(text, str):
        return False

    if re.search(r".*[A-Za-z]+.*", text):
        return True
    return False


def detect_only_english(text):
    """
    Detects if the given text contains only English characters.

    English characters are within the Unicode ranges A-Z and a-z.

    Parameters:
    text (str): The text to check.

    Returns:
    bool: True if the text contains only English characters, False otherwise.
    """
    if not isinstance(text, str):
        return False

    # Check if the text contains only English letters, spaces, and common punctuation
    if re.match(r'^[A-Za-z\s.,!?;:\'"-]+$', text):
        return True
    return False


def detect_french(text):
    if not isinstance(text, str):
        return False
    if re.search(r"[àâçéèêëîïôûùüÿæœ]", text, re.IGNORECASE):
        return True
    return False

def detect_arabic(text):
    if not isinstance(text, str):
        return False
    if re.search(r"[\u0600-\u06FF]", text):
        return True
    return False


def remove_only_brackets(text):
    if not isinstance(text, str):
        return None
    # 대괄호와 괄호만 제거하는 정규식
    #text = re.sub(r'[\[\]()]', '', text)  # 대괄호와 소괄호만 제거
    text = re.sub(r'\s*[\[\]()]\s*', '', text)

    return text



def detect_traditional_chinese(text):
    """
    Detects if the given text contains Traditional Chinese characters.

    Traditional Chinese characters are within the Unicode ranges \u4E00-\u9FFF, \u3400-\u4DBF, and \u20000-\u2A6DF.

    Parameters:
    text (str): The text to check.

    Returns:
    bool: True if the text contains Traditional Chinese characters, False otherwise.
    """
    if not isinstance(text, str):
        return None
    for char in text:
        if ('\u4E00' <= char <= '\u9FFF') or ('\u3400' <= char <= '\u4DBF') or ('\u20000' <= char <= '\u2A6DF'):
            return True
    return False

def src_process_and_save_data(main_df, languages):
    for lang in languages:
        df_lang = main_df[main_df['src_code'] == lang]
        
        with open(os.path.join(os.getcwd(), 'pickle_folder', f'src_{lang}_df.pkl'), 'wb') as f:
            pickle.dump(df_lang, f)
        
        print(f"Saved {lang} data with shape: {df_lang.shape}")

def dst_process_and_save_data(main_df, languages):
    for lang in languages:
        df_lang = main_df[main_df['dst_code'] == lang]
        
        with open(os.path.join(os.getcwd(), 'pickle_folder', f'dst_{lang}_df.pkl'), 'wb') as f:
            pickle.dump(df_lang, f)
        
        print(f"Saved {lang} data with shape: {df_lang.shape}")


def src_load_data(languages):
    dfs = {}
    for lang in languages:
        with open(os.path.join(os.getcwd(), 'pickle_folder', f'src_{lang}_df.pkl'), 'rb') as f:
            dfs[lang] = pickle.load(f)
            print(f"Loaded SRC {lang} data with shape: {dfs[lang].shape}")
    
    # Dynamically create variables for each DataFrame
    for lang in languages:
        globals()[f"{lang.lower().replace(' ', '_')}_df"] = dfs[lang]
    
    return dfs



def dst_load_data(languages):
    dfs = {}
    for lang in languages:
        with open(os.path.join(os.getcwd(), 'pickle_folder', f'dst_{lang}_df.pkl'), 'rb') as f:
            dfs[lang] = pickle.load(f)
            print(f"Loaded DST {lang} data with shape: {dfs[lang].shape}")
    
    # Dynamically create variables for each DataFrame
    for lang in languages:
        globals()[f"{lang.lower().replace(' ', '_')}_df"] = dfs[lang]
    
    return dfs
    

    
def preprocess_data(main_df):

    
    # main_df['src_content'] = main_df['src_content'].str.replace(r'[()]', '', regex=True)
    main_df['src_content'] = main_df['src_content'].apply(remove_only_brackets)
    main_df['dst_content'] = main_df['dst_content'].apply(remove_only_brackets)

    main_df['src_content'] = main_df['src_content'].apply(replace_newline_with_space)
    main_df['dst_content'] = main_df['dst_content'].apply(replace_newline_with_space)

    src_dst_contains_same_value = main_df[
        (main_df['src_content'] == main_df['dst_content'])
    ]
    
    src_dst_code_same_value_df = main_df[
        (main_df['dst_code'] == main_df['src_code'])
    ]

    rows_with_digits = main_df[
        (main_df['src_content'].apply(is_numeric_string)) &
        (main_df['dst_content'].apply(is_numeric_string))
    ]

    special_char_only = main_df[
        (main_df['src_content'].apply(lambda x: contains_special_character(x) and len(x) == 1)) |
        (main_df['dst_content'].apply(lambda x: contains_special_character(x) and len(x) == 1))
    ]

    is_not_string_instance = main_df[
        ~main_df['src_content'].apply(is_string_instance) |
        ~main_df['dst_content'].apply(is_string_instance)
    ]

    korean_jamo_only = main_df[
        main_df['src_content'].apply(lambda x: detect_korean_jamo(x) == 'jamo' and len(x) == 1) |
        main_df['dst_content'].apply(lambda x: detect_korean_jamo(x) == 'jamo' and len(x) == 1)
    ]

    # 특수 문자만 포함된 행을 제외
    main_df = main_df[~main_df.index.isin(src_dst_contains_same_value.index)]
    main_df = main_df[~main_df.index.isin(src_dst_code_same_value_df.index)]
    main_df = main_df[~main_df.index.isin(rows_with_digits.index)]
    main_df = main_df[~main_df.index.isin(special_char_only.index)] 
    main_df = main_df[~main_df.index.isin(is_not_string_instance.index)]
    main_df = main_df[~main_df.index.isin(korean_jamo_only.index)]
    
    
    
    return main_df


def concat_dataframes(df1, df2):
    print(f"Successfully stored {len(df2)} rows")
    return pd.concat([df1, df2], axis=0, ignore_index=True)


def src_data_analysis(main_df: pd.DataFrame, loaded_dfs: dict) -> pd.DataFrame:
    # Create an empty DataFrame
    src_lang_fix_df = pd.DataFrame()
    print("src_lang_fix_df")

    
    arabic_df = loaded_dfs['Arabic']
    korean_df = loaded_dfs['Korean']
    japanese_df = loaded_dfs['Japanese']
    english_df = loaded_dfs['English']
    traditional_chinese_df = loaded_dfs['Chinese (Traditional)']
    simplified_chinese_df = loaded_dfs['Chinese (Simplified)']
    french_df = loaded_dfs['French']
    spanish_df = loaded_dfs['Spanish']
    italian_df = loaded_dfs['Italian']
    indonesian_df = loaded_dfs['Indonesian']

    
    arabic_df['src_is_arabic'] = arabic_df['src_content'].apply(detect_arabic)
    arabic_df['src_is_english'] = arabic_df['src_content'].apply(detect_english)

    
    filter_df = arabic_df[
        (arabic_df['src_is_arabic'] == False) &
        (arabic_df['src_is_english'] == False)
    ]
    
    src_lang_fix_df = concat_dataframes(src_lang_fix_df, filter_df)

    filter_df = pd.DataFrame()
   
    english_df['src_is_english'] = english_df['src_content'].apply(detect_english)
    english_df['src_contains_numbers'] = english_df['src_content'].apply(is_numeric_string)

    filter_df = english_df[
        (english_df['src_is_english'] == False) &
        (english_df['src_contains_numbers'] == True)
    ]

    src_lang_fix_df = concat_dataframes(src_lang_fix_df, filter_df)
    

    filter_df = pd.DataFrame()
    
    
    korean_df['src_is_korean'] = korean_df['src_content'].apply(detect_korean)
    korean_df['src_is_chinese'] = korean_df['src_content'].apply(detect_chinese)
    korean_df['fasttext_predict_lang'] = korean_df['src_content'].apply(fast_predict_language)

    
    filter_df = korean_df[
        (korean_df['src_is_korean'] == False) & 
        (korean_df['src_is_chinese'] == False) & 
        (korean_df['fasttext_predict_lang'] != 'ko')
    ]
    src_lang_fix_df = concat_dataframes(src_lang_fix_df, filter_df)

    filter_df = pd.DataFrame()

    traditional_chinese_df['src_is_chinese'] = traditional_chinese_df['src_content'].apply(detect_chinese)

    filter_df = traditional_chinese_df[
        (traditional_chinese_df['src_is_chinese'] == False)  
    ]

    src_lang_fix_df = concat_dataframes(src_lang_fix_df, filter_df)


    filter_df = pd.DataFrame()
    
    simplified_chinese_df['src_is_chinese'] = simplified_chinese_df['src_content'].apply(detect_chinese)
    
    filter_df = simplified_chinese_df[
        (simplified_chinese_df['src_is_chinese'] == False) 
    ]

    src_lang_fix_df = concat_dataframes(src_lang_fix_df, filter_df)

    
    filter_df = pd.DataFrame()
    
    
    french_df['src_is_french'] = french_df['src_content'].apply(detect_french)    
    french_df['src_is_english'] = french_df['src_content'].apply(detect_english)

    filter_df = french_df[
        (french_df['src_is_french'] == False) &
        (french_df['src_is_english'] == False) 
    ]

    src_lang_fix_df = concat_dataframes(src_lang_fix_df, filter_df)

    filter_df = pd.DataFrame()
    


    return src_lang_fix_df



def dst_data_analysis(main_df, d_loaded_dfs):
    dst_lang_fix_df = pd.DataFrame()
    print("dst_lang_fix_df")
    
    korean_df = d_loaded_dfs['Korean']
    
    
    filter_df = korean_df[
        (korean_df['dst_content'].apply(detect_only_korean) == False) &
        (korean_df['dst_content'].apply(detect_korean) == False) &
        (korean_df['dst_content'].str.contains('\d') == False)
    ]
    
    dst_lang_fix_df = concat_dataframes(dst_lang_fix_df, filter_df)

    filter_df = pd.DataFrame()
    
    #korean_df.to_csv('dst_korean_df.csv', index=False)
    
    
    english_df = d_loaded_dfs['English']

    
    filter_df = english_df[
        (english_df['dst_content'].apply(detect_english) == False) &
        (english_df['dst_content'].str.contains('\d') == False)
    ]

    dst_lang_fix_df = concat_dataframes(dst_lang_fix_df, filter_df)
    
    filter_df = pd.DataFrame()
    
    #english_df.to_csv('dst_english_df.csv', index=False)
    
    hindi_df = d_loaded_dfs['Hindi']
    #hindi_df.to_csv('dst_hindi_df.csv', index=False)
    
    japanese_df = d_loaded_dfs['Japanese']

    filter_df = japanese_df[
        (japanese_df['dst_content'].apply(detect_only_korean) == True) &
        (japanese_df['dst_content'].str.contains('\d') == False)
    ]

    dst_lang_fix_df = concat_dataframes(dst_lang_fix_df, filter_df)
    filter_df = pd.DataFrame()
  
   # japanese_df.to_csv('dst_japanese_df.csv', index=False)
    

    # russian_df = d_loaded_dfs['Russian']
    
    # russian_df = russian_df[
    #     russian_df['dst_content'].apply(fast_predict_language) != 'ru'
    # ]
    
    # dst_lang_fix_df = concat_dataframes(dst_lang_fix_df, russian_df)
    
    # russian_df.to_csv('dst_russian_df.csv', index=False)
    
    # french_df = d_loaded_dfs['French']

    # french_df = french_df[
    #     french_df['dst_content'].apply(detect_only_korean) |
    #     french_df['dst_content'].apply(detect_only_chinese) |
    #     french_df['dst_content'].apply(detect_only_english) 
    # ]
        
    
    # french_df.to_csv('dst_french_df.csv', index=False)
    
    # spanish_df = d_loaded_dfs['Spanish']

    # spanish_df = spanish_df[
    #     spanish_df['dst_content'].apply(detect_only_korean) |
    #     spanish_df['dst_content'].apply(detect_only_chinese) 
    #     #spanish_df['dst_content'].apply(detect_only_english) 
    # ]
    
    
    # spanish_df.to_csv('dst_spanish_df.csv', index=False)
    
    # german_df = d_loaded_dfs['German']
    
    
    # german_df = german_df[
    #     german_df['dst_content'].apply(detect_only_korean) |
    #     german_df['dst_content'].apply(detect_only_chinese) 
    #     #german_df['dst_content'].apply(detect_only_english) 
    # ]
    
    # german_df.to_csv('dst_german_df.csv', index=False)
    
    # indonesian_df = d_loaded_dfs['Indonesian']

    # indonesian_df = indonesian_df[
    #     indonesian_df['dst_content'].apply(detect_only_korean) |
    #     indonesian_df['dst_content'].apply(detect_only_chinese) 
    # #    indonesian_df['dst_content'].apply(detect_only_english) 
    # ]
    
    
    # indonesian_df.to_csv('dst_indonesian_df.csv', index=False)
    
    # italian_df = d_loaded_dfs['Italian']

    # italian_df = italian_df[
    #     italian_df['dst_content'].apply(detect_only_korean) |
    #     italian_df['dst_content'].apply(detect_only_chinese) 
    # #   italian_df['dst_content'].apply(detect_only_english) 
    # ]
    
    # italian_df.to_csv('dst_italian_df.csv', index=False)
    
    # malay_df = d_loaded_dfs['Malay']

    # malay_df = malay_df[
    #     malay_df['dst_content'].apply(detect_only_korean) |
    #     malay_df['dst_content'].apply(detect_only_chinese) 
    #   # malay_df['dst_content'].apply(detect_only_english) 
    # ]
    
    
    # malay_df.to_csv('dst_malay_df.csv', index=False)
    
    # thai_df = d_loaded_dfs['Thai']
    
    # thai_df = thai_df[
    #     thai_df['dst_content'].apply(detect_only_korean) |
    #     thai_df['dst_content'].apply(detect_only_chinese) |
    #     thai_df['dst_content'].apply(detect_only_english) 
    # ]
    # thai_df.to_csv('dst_thai_df.csv', index=False)
    
    # vietnamese_df = d_loaded_dfs['Vietnamese']
    
    # vietnamese_df = vietnamese_df[
    #     vietnamese_df['dst_content'].apply(detect_only_korean) |
    #     vietnamese_df['dst_content'].apply(detect_only_chinese) |
    #     vietnamese_df['dst_content'].apply(detect_only_english) 
    # ]
    
    
    # vietnamese_df.to_csv('dst_vietnamese_df.csv', index=False)
    
    # arabic_df = d_loaded_dfs['Arabic']
    
    # arabic_df = arabic_df[
    #     arabic_df['dst_content'].apply(detect_only_korean) |
    #     arabic_df['dst_content'].apply(detect_only_chinese) |
    #     arabic_df['dst_content'].apply(detect_only_english) 
    # ]
    
    
    # arabic_df.to_csv('dst_arabic_df.csv', index=False)
    
    # mongolian_df = d_loaded_dfs['Mongolian']
    
    # mongolian_df = mongolian_df[
    #     mongolian_df['dst_content'].apply(detect_only_korean) |
    #     mongolian_df['dst_content'].apply(detect_only_chinese) |
    #     mongolian_df['dst_content'].apply(detect_only_english) 
    # ]
    
    # mongolian_df.to_csv('dst_mongolian_df.csv', index=False)
    
    # polish_df = d_loaded_dfs['Polish']
    
    # polish_df = polish_df[
    #     polish_df['dst_content'].apply(detect_only_korean) |
    #     polish_df['dst_content'].apply(detect_only_chinese) |
    #     polish_df['dst_content'].apply(detect_only_english) 
    # ]
    
    
    # polish_df.to_csv('dst_polish_df.csv', index=False)
    
    # simplified_chinese_df = d_loaded_dfs['Chinese (Simplified)']
    
    # simplified_chinese_df = simplified_chinese_df[
    #     simplified_chinese_df['dst_content'].apply(detect_only_korean) |
    #     simplified_chinese_df['dst_content'].apply(detect_only_english) 
    # ]
    
    
    # simplified_chinese_df.to_csv('dst_simplified_chinese_df.csv', index=False)
    
    # traditional_chinese_df = d_loaded_dfs['Chinese (Traditional)']
    
    # traditional_chinese_df = traditional_chinese_df[
    #     traditional_chinese_df['dst_content'].apply(detect_only_korean) |
    #     traditional_chinese_df['dst_content'].apply(detect_only_english) 
    # ]
    
    # traditional_chinese_df.to_csv('dst_traditional_chinese_df.csv', index=False)

    return dst_lang_fix_df





def main():
    # Load the Excel file
    file_path = 'place_tm_20240723_083721.xlsx'
    main_df = pd.read_excel(file_path)

    # Display the first few rows of the dataframe
    print(main_df.info())
    print("\n")

    src_languages = main_df['src_code'].unique().tolist()
    dst_languages = main_df['dst_code'].unique().tolist()


    # Preprocess data
    main_df = preprocess_data(main_df)

    src_process_and_save_data(main_df, src_languages) 
    dst_process_and_save_data(main_df, dst_languages)
    
    print("\n")
    # Load data
    src_loaded_dfs = src_load_data(src_languages) 
    dst_loaded_dfs = dst_load_data(dst_languages)

    src_lang_fix_df = src_data_analysis(main_df, src_loaded_dfs)
    dst_lang_fix_df = dst_data_analysis(main_df, dst_loaded_dfs)

    src_lang_fix_df[['src_code', 'src_content', 'dst_code', 'dst_content']].to_csv('src_lang_fix_df.csv', index=False)
    dst_lang_fix_df[['src_code', 'src_content', 'dst_code', 'dst_content']].to_csv('dst_lang_fix_df.csv', index=False)

    

if __name__ == "__main__":
    main()