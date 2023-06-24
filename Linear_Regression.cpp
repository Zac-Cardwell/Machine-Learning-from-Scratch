//============================================================================
// Name        : test1.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <vector>
#include <cmath>


class regression{
public:
	std::vector<double> train_x;
	std::vector<double> train_y;
	int size = 0;
	regression(std::vector<double> x, std::vector<double> y){
		train_x = x;
		train_y = y;
		size = double(std::size(y));
	}

	void train(int epoch, double lr);
	double pred(double x);
	void backpass(std::vector<double> yhat, double lr);

private :
	double weight = 0.0;
	double bias = 0.0;
};



void regression::train(int epoch, double lr = .001){
	std::vector<double> yhat;
	double MSE = 0.0;
	for(int i=0; i<epoch; i++){
		yhat.clear();
		for(int j=0; j < size; j++){
			yhat.push_back((weight * train_x[j]) + bias);
		}
		std::cout<<yhat[0] << std::endl;
		for(int k=0; k<size; k++){
			MSE += std::pow((train_y[k] - yhat[k]), 2);
		}
		MSE = MSE/size;
		std::cout<< "epoch: "<< i << " Error: "<< MSE<<std::endl;
		backpass(yhat, lr);
	}
}

void regression::backpass(std::vector<double> yhat, double lr){
	double dervM = 0.0;
	double dervB = 0.0;
	for(int i =0; i<size; i++){
		dervM += (train_x[i]*(yhat[i] - train_y[i]));
		dervB += (yhat[i] - train_y[i]);
	}
	std::cout<<dervM<<std::endl;
	dervM *= (-2.0/size);
	dervB *= (-1.0/size);
	weight += dervM*lr;
	bias += dervB*lr;
}

double regression::pred(double x){
	return weight*x + bias;
}

int main() {

	std::vector<double> x;
	std::vector<double> y;
	for(int i=1; i<=50; i++){
		x.push_back(i);
		y.push_back(.5*i);
	}
	regression test(x, y);
	test.train(500, .00005);
	std::cout<< test.pred(5);

	return 0;
}
