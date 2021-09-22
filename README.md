# AutoRCCar
##Python3 + OpenCV + YOLOv3 tiny

Click on picture to see self-driving in action
</br>
<a href="https://www.youtube.com/watch?v=FiEcJnCTDOk
" target="_blank"><img src="https://lh3.googleusercontent.com/aGGLK0x0jdoS7XynmQpQBRLGTo959-C4PEDoKnMwxBQlQ-AlL_PAbzwIFh344U_x0ddY7UwCqeQpUU9NFKojy0BW7Prr0IGLU5kBToK3Hzm_MATvn9cnR639olNGzdPXViybp4r1H5b0iwjETr-OxHhBfhkmpCF6Hqs-yrEj6BW_GNMn8c-tWA6iUE3NvmKIh2DyNpjct-LosTBKY6uvvEAt5SDy9H4LXg2aZjMG5SLzNyDS02IHtqzL7hmD5uKu7gfzlSMKanOr-s1xA2TosOvY2PqZlkMz-3kZQYFnoOGlrzaekCKl0vgTgiAUeptwjpPOomYmNPNY9ao9YKXtUPwBdf2EqOiGOjp2uRTC6P5oHw7Mh96LCOFLuM_U7nqnwTNPQWddYrGm7iFXmjXmEDuo8xo2ttATyGB8FZgWgZ9Ga6fn98mzAyyipAjsbTAUVGlaEnK_b4676BAa-CtKiS7FpnbkmFpn-ED97fIjDy6mF9ctEjR1agvjrZ3e8_56lDUNIJo4D8_GYTAV0JkGDRZLgehSROLnu02zvDBi16wtJGusW-2n45XnVDcSeGbuydzSWZS5XQWisKifDZmj9cCKtlQg8JjXHQyQpDw2XyH3QZqnGH1wx1BhOwVFpUKCeKG8dZM2Ph40J9b0EVJtrxsJC-3M004Gl89DCLeOZbl8pnX0Vi-g2axX_2PMyC8HsBDnRhi8v-6N1lXR72HmES4l=w502-h470-no?authuser=0" width="360" height="240" border="10" /></a>

This project builds a self-driving RC car using Jetson Nano and open source software. Jetson Nano collects inputs from a camera module,Jetson Nano processes input images for object detection (trafic sign,direction of line,intersection). A neural network model runs on Jetson Nano and makes predictions combined with some images processing program for steering based on input images. 

##Data Collection
</br>
<img src="https://lh3.googleusercontent.com/HQKPdHileuAmyoBXMzMSXxkVAbKabhZUuwNmzNbTlLPt37aMVaFhSIWJTZv12ju2dkmCr22OuXJrCAgwwkwx5kTdI4nRXggHQqCv3ZN5lBYJYyGyRaLrPT92wZ_jcwMC-PL0hE-KWU1d04Mhl-pEpQLQTztSZVSd3Xm8xGMLdyU44ws3C_h9GY1poi3SQC1aKpZdW4PiwiVLLrF5o12S3jDB66sWR8sAE34n2jJnBaBD2_fpKkRtAtAOhC22X0QiE5ZChg4O-U_dh7mFef83Oz8a9XfXduW4dKerZQN5DlyVDoUVojst1vosl08-l3WGOh9MFkk7N9g-NTs1JI4QFP4zopYxFLe6aQ7E90Spq7qVBWbGWl-kArn_2a0tHtpDPVig1EFzCVSAkfH7CgYOT4WY7Y_71ehSdRP36rdAhyL6hhy9GwjHTtCK2_Bs7iGcKeJweVl1_hVFvGVaqwDWG1g39QFSwV6RvN9bnA0FCC_VdXDP6NLFfU35LLzJkNdJEEEwCFYjWdH-g-8UZMjcHV3e6CbPZxARLZThbziGyIj9zQiz3qxulmyKbvlfHPVUT6J8_uJtoHZRAXJOb0qfAF5eIItdeieSjXiYvKuZbjhQVBRo4CRXJW1QOWYExksd6jPy2ihmP1lE6mjsbAOu311nwq2YkeuWKVPp1pYMc82GOeKrXuIdf4ZdLnjNnEc8EWBuAmK2DdR69XPL60rCtaoE=w1366-h768-no?authuser=0" width="600">

Using webcam to take images and label by LABELIMG
In this project we have 4 classes
</br>
<img src="https://lh3.googleusercontent.com/2TpOfgfPaCp37Svvx6rOhKEpeDWwUMHPywBVVFA3qyMR4bpHZMzVr7Wxq-AIAn_aQSp6_QvFXa6fCpFI5dQlzZGmUVotRENZO0Km8zoeTQhZwlLsJE1uzZDxCmJFI2WNO8nijRupDz-bRUvinc0jmwPNNTfzW465btihJOuEeSYzzGH_ipf-iv-ivt4SGkxW9E9vAMIjur2tSiC1nA-qMYJZ04Ugk6VnLVaCRvQ5YCSPZLIXisoluoo8PWto53kNax5DaGXLX-oF7iP5cyuEkpDxm99pSUst_D5uJNg53FNOYs_OMJ_GYdxWWTDjMpVleo7KlyS7R0vVYSedqHOGsCacxBixBJ0VHnr99uf92w6w4zOcnxsqTJNRCiTj2tC7Quz0eHMYVCxlwzKUdDCB-MFyXwkfCdnnoMpXmj5vs_fPzeoKtsJs9o9GaKLb8F-0aG55g72YNohr1VG2T-lEsnur6eSr1unDPFBtPenBeuPKeyP4WdhtxlUtRZR_GArN3e6JPy0hz59dssLpo6R4VURfCA4YEcp2MIignNIjlmdJ4-Nxnryw5IjAO9YF2jfV6p7R6WwXdnGmlROHWoc20cGto5janPb0hLdw80jLrjRVnMXZ5hcjQWyD4I0j3PNkmwJgp3Os8IcnVH0DyIBh6sBNtzwKW8FFVyAwcYju7Aw0oc5E-LCedQUTw4RvKRm-YUxG5eRzdH2SYy1pZtNpOT5k=w439-h114-no?authuser=0" width="600">
