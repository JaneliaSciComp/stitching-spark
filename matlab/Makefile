MATLAB_ROOT=/Applications/Matlab/MATLAB_R2023a.app
MCC=${MATLAB_ROOT}/bin/mcc

.PHONY: build
build:
	-mkdir -p bin
	${MCC} -v \
	  -R '-nojvm' \
	  -m \
	  -d bin \
	  matlab_decon.m \
	 -I .

.PHONY: clean
clean:
	-rm -rf bin
