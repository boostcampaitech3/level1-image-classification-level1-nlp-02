import os

# T3116 신규범님 
def define_age(age):
    if age<30:
        return 0
    elif age<60:
        return 1
    else:
        return 2

def define_mask(mask_type):
    if mask_type == 'normal':
        mask = 2 
    elif mask_type == 'incorrect_mask':
        mask = 1
    else:
        mask = 0

    return mask

def processing_df(df, config):
    df['image_path'] = df['path'].map(lambda x: [i for i in os.listdir(os.path.join(config['dir']['image_dir'].format('train') ,x)) if not i.startswith('.')]) # 상세 path list 만들기
    df = df.explode('image_path') # path list row 단위로 나누기
    df['detail_path'] = df['path'] + '/' + df['image_path'] # image 위치 부여
    df['mask_type'] = df['image_path'].map(lambda x:x.split('.')[0]) # mask type 부여
    df['age_label'] = df['age'].apply(define_age) # age 구간화
    df['mask_label'] = df['mask_type'].map(define_mask) # mask type labeling
    df['gender_label'] = df['gender'].map(lambda x: 0 if x == 'male' else 1) # gender labeling
    df['label'] = df['mask_label']*6 + df['gender_label']*3 + df['age_label'] # class labeling
    df = df[['id','gender','race','age','mask_type','path','image_path','detail_path','age_label','mask_label','gender_label','label']] # 필요한 col만 추출
    df = df.reset_index(drop=True)

    return df