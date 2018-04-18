#pragma once
#include <windows.h>  

class CSerial
{
public:
	CSerial(void);
	~CSerial(void);

	//�򿪴���  
	BOOL OpenSerialPort(TCHAR* port, UINT baud_rate, BYTE date_bits, BYTE stop_bit, BYTE parity = NOPARITY);

	//��������  
	BOOL SendData(unsigned char* data, int len);
public:
	HANDLE m_hComm;

//////////////////////new/////////////////////////
public:
	int Serial_Send_Yaw(float value);
	int Serial_Send_Pitch(float value);
};