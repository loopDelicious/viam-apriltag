bundle:
	tar -czf module.tar.gz *.sh src requirements.txt

upload:
	viam module upload --version $(version) --platform $(platform) module.tar.gz

clean:
	rm -rf module.tar.gz