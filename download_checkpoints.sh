wget https://www.dropbox.com/s/anjzft8njjk3rh6/Iter29000_model_m_75.6006_f_0.8330_c_0.7377_r_0.3380.pt?dl=1 -O ./Iter29000_model_m_75.6006_f_0.8330_c_0.7377_r_0.3380.pt
wget https://www.dropbox.com/s/zq8rkh3nlsj20r3/Iter31500_model.pt?dl=1 -O ./Iter31500_model.pt

mkdir -p ./our_trained_ckpts/
mv ./Iter29000_model_m_75.6006_f_0.8330_c_0.7377_r_0.3380.pt ./our_trained_ckpts/Iter29000_model_m_75.6006_f_0.8330_c_0.7377_r_0.3380.pt
mv ./Iter31500_model.pt ./our_trained_ckpts/Iter31500_model.pt