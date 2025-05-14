download_data:
	wget -O dataset/caltech https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip

unzip:
	unzip dataset/caltech-101.zip -d dataset/caltech 
