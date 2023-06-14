import numpy as np
# import os
# import imageio as iio
# from PIL import Image


class HanjaData:    
    def get_data(self, path, f_name, flag=False):
        if flag:
            return np.load(f'{f_name}.npz')
        return np.load(f'{path}/{f_name}.npz')

    # def to_npz(path, num):
    #     global hanja_data
    #     all_file_path = [os.path.join(r, file) for r,d,f in os.walk(path) for file in f]
    #     changed_img = [np.asarray(Image.open(filename).resize((num,num))) for filename in all_file_path]
    #     resized_img = np.array(changed_img)
    #     to_labels =  [os.path.join(r, file).split('/')[-1][:5] for r,d,f in os.walk(path) for file in f]
        
    #     new_path = '/'.join(path.split('/')[:-2])+'/'
    #     np.savez(new_path+'hanja_data.npz', images = resized_img, labels = to_labels)
    #     hanja_data = np.load(new_path+'hanja_data.npz')
    #     print(f"npz file saved on: {new_path}")
    #     return hanja_data
        
    def X_Y_split(examples, labels, train_frac, random_num):
        assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"
        global X_train, X_val, X_test,  Y_train, Y_val, Y_test

        X_train, X_tmp, Y_train, Y_tmp = train_test_split(
                                            examples, labels, train_size=train_frac, random_state=random_num)

        X_val, X_test, Y_val, Y_test   = train_test_split(
                                            X_tmp, Y_tmp, train_size=0.5, random_state=random_num)
        print(f"X, Y split >> train({train_frac}) : val({(1-train_frac)/2:.2f}) : test({(1-train_frac)/2:.2f})")
        return X_train, X_val, X_test,  Y_train, Y_val, Y_test