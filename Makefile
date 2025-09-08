ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))


SRCDIR=${ROOTDIR}/emg_experiment_simple
INSTALL_LOG_FILE=${ROOTDIR}/install.log
VENV_SUBDIR=${ROOTDIR}/venv
DATAFILE=${ROOTDIR}/MK_10_03_2022_EMG.tar.xz
DATAFILE2=${ROOTDIR}/KrzysztofJ_all_EMG.tar.xz
DATAFILEID=102kCvTh_qK8ajqnSNz3VE6ffZKfwcKnf
DATAFILEID2=1GL_MIj2OsdjUbsehWN2tZdAn3kvYVK71
DATADIR=${ROOTDIR}/data


PYTHON=python
SYSPYTHON=python
PIP=pip
TAR := $(shell command -v gtar >/dev/null 2>&1 && echo gtar || echo tar)
CURL=curl

LOGDIR=${ROOTDIR}/testlogs
LOGFILE=${LOGDIR}/`date +'%y-%m-%d_%H-%M-%S'`.log

VENV_OPTIONS=

ifeq ($(OS),Windows_NT)
	ACTIVATE:=. ${VENV_SUBDIR}/Scripts/activate
else
	ACTIVATE:=. ${VENV_SUBDIR}/bin/activate
endif


.PHONY: all clean test docs

all: experiment

clean:
	rm -rf ${VENV_SUBDIR}

venv:
	${SYSPYTHON} -m venv --upgrade-deps ${VENV_OPTIONS} ${VENV_SUBDIR}
	${ACTIVATE}; ${PYTHON} -m ${PIP} install -e ${ROOTDIR} --prefer-binary --log ${INSTALL_LOG_FILE}

experiment: venv prepare_data
	@echo "Experiment"
	${ACTIVATE}; experiment

prepare_data: $(DATADIR)
	@echo "Prepare data"

feature_extraction: venv prepare_data
	@echo "Feature extraction"
	${ACTIVATE}; feature_extraction

$(DATAFILE):
	${CURL} -L -o ${DATAFILE} "https://drive.usercontent.google.com/download?id=${DATAFILEID}&export=download&authuser=1&confirm=t"

$(DATAFILE2):
	${CURL} -L -o ${DATAFILE2} "https://drive.usercontent.google.com/download?id=${DATAFILEID2}&export=download&authuser=1&confirm=t"

$(DATADIR): $(DATAFILE) $(DATAFILE2)
	mkdir -p ${DATADIR}
	${TAR} -xvf ${DATAFILE} --directory ${DATADIR}
	${TAR} -xvf ${DATAFILE2} --directory ${DATADIR}