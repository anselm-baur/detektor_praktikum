#!/bin/bash


BOARD_UP="m4587"
BOARD_DOWN="m4520"

LOG_UP="log_${BOARD_UP}"
LOG_DOWN="log_${BOARD_DOWN}"

if [ -f $LOG_UP ]; then
	rm $LOG_UP
fi

if [ -f $LOG_DOWN ]; then
	rm $LOG_DOWN
fi


for ((NUM_MEASURE=0; NUM_MEASURE<3; NUM_MEASURE++))
	PXAR_UP="cat xray | ./pxar/bin/pXar -d pxar/output/${BOARD_UP}/ -T 35 -r rootfiles/measuring/meas_${NUM_MEASURE}.root"
	PXAR_DOWN="cat xray | ./pxar/bin/pXar -d pxar/output/${BOARD_UP}/ -T 35 -r rootfiles/measuring/meas_${NUM_MEASURE}.root"
	do
		stdbuf -o0 ${PXAR_UP} > $LOG_UP &
		pid_1=$!

		stdbuf -o0 ${PXAR_DOWN} > $LOG_DOWN &
		pid_2=$!
		
		echo ""
		echo "###################################################################"
		echo "            RESTARTING SCRIPT - RUN ${NUM_MEASURE}"
		echo "###################################################################"

		echo "Starting Board ${BOARD_UP} @PID: $pid_1           Starting Board ${BOARD_DOWN} @PID: $pid_2"
		while ps $pid_1>/dev/null || ps $pid_2>/dev/null; do
			OUTPUT_UP=""
			OUTPUT_DOWN=""

			OUTPUT_UP=$(tail -2 $LOG_UP | head -1 | grep DEBUGAPI)
			if [ -z $OUTPUT_UP ]; then
				OUTPUT_UP=$(tail -2 $LOG_UP | head -1)
			fi

			OUTPUT_DOWN=$(tail -2 $LOG_DOWN | head -1 | grep DEBUGAPI)
			if [ -z $OUTPUT_DOWN ]; then
				OUTPUT_DOWN=$(tail -2 $LOG_DOWN | head -1)
			fi

			echo "Board ${BOARD_UP}: $OUTPUT_UP         Board ${BOARD_DOWN}: $OUTPUT_DOWN"
			sleep 0.1
		done
	done
