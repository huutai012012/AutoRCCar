# Self-Driving Car
## Python3 + OpenCV + YOLOv3 tiny

<p style="color:Tomato;font-size:300%;">=>>Click on GIF to see self-driving in action<<=</p>
</br>
<a href="https://www.youtube.com/watch?v=FiEcJnCTDOk
" target="_blank"><img src="https://lh3.googleusercontent.com/TYOKntBHSvbR1lsVwdSzKE4ai13_t8TwKHAbUwjIz_nInBVqfOuIOFeb7SvtyUaNH8CNSY7UZpxTd82BTq-LqY0fgVLKYqoGZKONpSF67vAnOjez4177sReNWpDDZmqKVSrJ4CgzmJ6Zy9fCT246BFb0arfwXsSnjiib_QcprMvz6osTJMdBNQ8Mu1FHIZxPFGp7fLkbocKqq35mrmjmqv8mmPvSFkQrHTlURpIqJDhLwPB3k4BSQ_kvfz8cIkEMmO94myNRKfztEBf1zQT_oiKyncdMOvRcaOS-KxbR80yBG9AGHw9s0z-HCXsmU4_JkjK5ia2la6lEB8Phr1-v2V6scuOVnl9BYNo80gDz5j0gw1FL9LSv6Ifk_722hdruV1D4vjOPBd2MEWdkrOOAYwrJIzjVT4kopvZcPGrpYBv7iyU7XN1Rg-4pX6HgIwTzgz6CwlZodevhDIS91H8YIEAUuKV_BNiTgUzsryH0nfOKRsxql2qOPOyd7dc5Epfvpglb51o84klCP5uXCCNgNNA6rOKnOibVlePFZ8DnxbiYN3-uwfWOcPChf5Q93lTh1RnhDi-6EuEEElyfqcihizZHYFdvVcA0uZ0lf1uHo9ygO5xzG5YPXg0P-lgN5R3MZU1w7SHlxE66k7M0-Uw_JZlbdgAqS_16zAb_joPa0GW9n2L-PbtltISHPwt3hyVG_Q-y19naFPomNk2zac27Jfdr=w328-h185-no?authuser=0" width="360" height="240" border="10" /></a>
</br>
<a href="https://www.youtube.com/watch?v=FiEcJnCTDOk
" target="_blank"><img src="https://photos.app.goo.gl/s9r9MShJt1zsYRFx9" width="360" height="240" border="10" /></a>
</br>
This project builds a self-driving car using Jetson Nano and open source software. Jetson Nano collects inputs from a camera module,Jetson Nano processes input images for object detection (trafic sign,direction of line,intersection). A neural network model runs on Jetson Nano and makes predictions combined with some images processing program for steering based on input images. 

## Data Collection
</br>
<img src="https://lh3.googleusercontent.com/HQKPdHileuAmyoBXMzMSXxkVAbKabhZUuwNmzNbTlLPt37aMVaFhSIWJTZv12ju2dkmCr22OuXJrCAgwwkwx5kTdI4nRXggHQqCv3ZN5lBYJYyGyRaLrPT92wZ_jcwMC-PL0hE-KWU1d04Mhl-pEpQLQTztSZVSd3Xm8xGMLdyU44ws3C_h9GY1poi3SQC1aKpZdW4PiwiVLLrF5o12S3jDB66sWR8sAE34n2jJnBaBD2_fpKkRtAtAOhC22X0QiE5ZChg4O-U_dh7mFef83Oz8a9XfXduW4dKerZQN5DlyVDoUVojst1vosl08-l3WGOh9MFkk7N9g-NTs1JI4QFP4zopYxFLe6aQ7E90Spq7qVBWbGWl-kArn_2a0tHtpDPVig1EFzCVSAkfH7CgYOT4WY7Y_71ehSdRP36rdAhyL6hhy9GwjHTtCK2_Bs7iGcKeJweVl1_hVFvGVaqwDWG1g39QFSwV6RvN9bnA0FCC_VdXDP6NLFfU35LLzJkNdJEEEwCFYjWdH-g-8UZMjcHV3e6CbPZxARLZThbziGyIj9zQiz3qxulmyKbvlfHPVUT6J8_uJtoHZRAXJOb0qfAF5eIItdeieSjXiYvKuZbjhQVBRo4CRXJW1QOWYExksd6jPy2ihmP1lE6mjsbAOu311nwq2YkeuWKVPp1pYMc82GOeKrXuIdf4ZdLnjNnEc8EWBuAmK2DdR69XPL60rCtaoE=w1366-h768-no?authuser=0" width="600">

Using webcam to take images and label by LABELIMG
In this project we have 4 classes
</br>
<img src="https://lh3.googleusercontent.com/2TpOfgfPaCp37Svvx6rOhKEpeDWwUMHPywBVVFA3qyMR4bpHZMzVr7Wxq-AIAn_aQSp6_QvFXa6fCpFI5dQlzZGmUVotRENZO0Km8zoeTQhZwlLsJE1uzZDxCmJFI2WNO8nijRupDz-bRUvinc0jmwPNNTfzW465btihJOuEeSYzzGH_ipf-iv-ivt4SGkxW9E9vAMIjur2tSiC1nA-qMYJZ04Ugk6VnLVaCRvQ5YCSPZLIXisoluoo8PWto53kNax5DaGXLX-oF7iP5cyuEkpDxm99pSUst_D5uJNg53FNOYs_OMJ_GYdxWWTDjMpVleo7KlyS7R0vVYSedqHOGsCacxBixBJ0VHnr99uf92w6w4zOcnxsqTJNRCiTj2tC7Quz0eHMYVCxlwzKUdDCB-MFyXwkfCdnnoMpXmj5vs_fPzeoKtsJs9o9GaKLb8F-0aG55g72YNohr1VG2T-lEsnur6eSr1unDPFBtPenBeuPKeyP4WdhtxlUtRZR_GArN3e6JPy0hz59dssLpo6R4VURfCA4YEcp2MIignNIjlmdJ4-Nxnryw5IjAO9YF2jfV6p7R6WwXdnGmlROHWoc20cGto5janPb0hLdw80jLrjRVnMXZ5hcjQWyD4I0j3PNkmwJgp3Os8IcnVH0DyIBh6sBNtzwKW8FFVyAwcYju7Aw0oc5E-LCedQUTw4RvKRm-YUxG5eRzdH2SYy1pZtNpOT5k=w439-h114-no?authuser=0" width="600">
</br>
## References
https://github.com/geek1111/Self-Driving-Car
https://github.com/hamuchiwa/AutoRCCar