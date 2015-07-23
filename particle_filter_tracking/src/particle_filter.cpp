
//  Created by Chaudhary Krishneel on 4/9/14.
//  Copyright (c) 2014 Chaudhary Krishneel. All rights reserved.

#include <particle_filter_tracking/particle_filter.h>

ParticleFilter::ParticleFilter() : threads_(4) {
   
}

cv::Mat ParticleFilter::state_transition() {
    Particle p;
    double dynamics[NUM_STATE][NUM_STATE] = {{1, 0, 1, 0},
                                             {0, 1, 0, 1},
                                             {0, 0, 1, 0},
                                             {0, 0, 0, 1}};
    cv::Mat dynamic_model = cv::Mat(NUM_STATE, NUM_STATE, CV_64F);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int j = 0; j < NUM_STATE; j++) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
       for (int i = 0; i < NUM_STATE; i++) {
          dynamic_model.at<double>(j, i) = dynamics[j][i];
       }
    }
    return dynamic_model;
}

std::vector<Particle> ParticleFilter::initialize_particles(
    cv::RNG &rng, double lowerX, double lowerY,
    double upperX, double upperY) {
    std::vector<Particle> p;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(threads_)
// #endif
    for (int i = 0; i < NUM_PARTICLES; i++) {
        Particle part__;
        part__.x = static_cast<double>(rng.uniform(lowerX, upperX));
        part__.y = static_cast<double>(rng.uniform(lowerY, upperY));
        
        part__.dx = static_cast<double>(rng.uniform(0.0, 2.0));
        part__.dy = static_cast<double>(rng.uniform(0.0, 2.0));
        p.push_back(part__);
    }
    return p;
}


std::vector<Particle> ParticleFilter::transition(
    std::vector<Particle> &p, cv::Mat &dynamics, cv::RNG &rng) {
    if (dynamics.rows != dynamics.cols &&
       dynamics.rows != NUM_STATE && dynamics.cols != NUM_STATE) {
       return std::vector<Particle>();
    }
    std::vector<Particle> transits;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(threads_)
// #endif
    for (int i = 0; i < NUM_PARTICLES; i++) {
       cv::Mat particleMD = cv::Mat(
          NUM_STATE, sizeof(char), CV_64F, &p[i]);
       cv::Mat trans = dynamics * particleMD;
        Particle t;
        t.x = static_cast<double>(trans.at<double>(0, 0)) +
           static_cast<double>(rng.gaussian(SIGMA));
        t.y = static_cast<double>(trans.at<double>(1, 0)) +
           static_cast<double>(rng.gaussian(SIGMA));
        t.dx = static_cast<double>(trans.at<double>(2, 0)) +
           static_cast<double>(rng.gaussian(SIGMA));
        t.dy = static_cast<double>(trans.at<double>(3, 0)) +
           static_cast<double>(rng.gaussian(SIGMA));
        transits.push_back(t);
    }
    return transits;
}

double ParticleFilter::evaluate_gaussian(
    double N, double z) {
    double val = static_cast<double>(z - N);
    return static_cast<double>((1.0/(sqrt(2.0*PI)) * SIGMA) *
                               exp(-((val * val) / (2*SIGMA*SIGMA))));
}

std::vector<double> ParticleFilter::normalizeWeight(
    std::vector<double> &z) {
    double sum = 0.0;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(threads_)
// #endif
    for (int i = 0; i < NUM_PARTICLES; i++) {
        sum += static_cast<double>(z[i]);
    }
    std::vector<double> w;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(threads_)
// #endif
    for (int i = 0; i < NUM_PARTICLES; i++) {
        w.push_back(static_cast<double>(z[i]/sum));
    }
    return w;
}

double ParticleFilter::gaussianNoise(
    double a, double b) {
    return rand() / (RAND_MAX + 1.0) * (b - a) + a;
}

std::vector<double> ParticleFilter::cumulativeSum(
    std::vector<double> &nWeights) {
    std::vector<double> cumsum;
    cumsum.push_back(nWeights[0]);
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(threads_)
// #endif
    for (int i = 1 ; i < NUM_PARTICLES; i++) {
        double c = nWeights[i] + cumsum[i-1];
        cumsum.push_back(c);
    }
    return cumsum;
}

void ParticleFilter::reSampling(
    std::vector<Particle> &x_P, std::vector<Particle> &xP_Update,
    std::vector<double> &nWeights) {
    std::vector<double> cumsum = cumulativeSum(nWeights);
    double s_ptX = static_cast<double>(
       this->gaussianNoise(0, 1.0/NUM_PARTICLES));
    int cdf_stX = 1;
    Particle p_arr[NUM_PARTICLES];
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int i = 0; i < NUM_PARTICLES; i++) {
        p_arr[i] = x_P[i];
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int j = 0; j < NUM_PARTICLES; j++) {
        double ptX = s_ptX + (1.0/NUM_PARTICLES)*(j-1);
        while (ptX > cumsum[cdf_stX]) {
            cdf_stX++;
        }
        p_arr[j] = xP_Update[cdf_stX];
    }
    x_P.clear();
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int i = 0; i < NUM_PARTICLES; i++) {
        x_P.push_back(p_arr[i]);
    }
}

Particle ParticleFilter::meanArr(
    std::vector<Particle> &xP_Update) {
    double sum = 0.0;
    double sumY = 0.0;
    double sumDX = 0.0;
    double sumDY = 0.0;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(threads_)
// #endif
    for (int i = 0; i < NUM_PARTICLES; i++) {
        sum     += static_cast<double>(xP_Update[i].x);
        sumY    += static_cast<double>(xP_Update[i].y);
        sumDX   += static_cast<double>(xP_Update[i].dx);
        sumDY   += static_cast<double>(xP_Update[i].dy);
    }
    Particle p;
    p.x      = static_cast<double>(sum/NUM_PARTICLES);
    p.y      = static_cast<double>(sumY/NUM_PARTICLES);
    p.dx     = static_cast<double>(sumDX/NUM_PARTICLES);
    p.dy     = static_cast<double>(sumDY/NUM_PARTICLES);
    return p;
}

void ParticleFilter::printParticles(
    cv::Mat &img, std::vector<Particle> &p) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
       cv::Point2f center = cv::Point2f(p[i].x, p[i].y);
       cv::circle(img, center, 3, cv::Scalar(0, 255, 255), CV_FILLED);
    }
}
