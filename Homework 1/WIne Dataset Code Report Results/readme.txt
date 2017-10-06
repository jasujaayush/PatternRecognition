To run the code execute following command :
python2.7 <compressor_name>_<classifier_name>.py
compressor_name : pca, lda
classifier_name : lda, nb
The number of samples can be altered in the code inside by setting the parameter n.
Also in the code, while compressing the data to required componets you might have to change the last parameter 
to the required value in the following api call : "<compressor_name>.compressor(eig_pairs, features, label, 1)"
