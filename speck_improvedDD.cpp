#include <iostream>
#include <cstdlib>
#include <random>
#include <vector>
#include <cstring>
#include <tuple>
#include <unordered_map>
#include <map>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>

using namespace std;

#define WORD_SIZE 16
#define BLOCK_SIZE (2 * (WORD_SIZE))
#define ALPHA 7
#define BETA 2
#define MASK_VAL 0xffff
#define MAX_ROUNDS 50
#define r0 (1.0 / (double)(1ULL << (uint64_t)BLOCK_SIZE))
#define TEST_DATA_logN (19)

inline uint32_t rol(uint32_t a, uint32_t b){
    uint32_t n = ((a << b) & MASK_VAL) | (a >> (WORD_SIZE - b));
    return(n);
}

inline uint32_t ror(uint32_t a, uint32_t b){
    uint32_t n = (a >> b) | (MASK_VAL & (a << (WORD_SIZE - b)));
    return(n);
}

inline void round_function(uint32_t a, uint32_t b, uint32_t k, uint32_t& x, uint32_t& y){
    uint32_t c0 = a; uint32_t c1 = b;
    c0 = ror(c0, ALPHA);
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA);
    c1 = c1 ^ c0;
    x = c0; y = c1;
}

inline void inverse_round_function(uint32_t a, uint32_t b, uint32_t k, uint32_t& x, uint32_t& y){
    uint32_t c0 = a; uint32_t c1 = b;
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA);
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA);
    x = c0; y = c1;
}

inline uint32_t decrypt_one_round(uint32_t c, uint32_t sk){
    uint32_t x,y;
    uint32_t c0 = c >> 16; uint32_t c1 = c & MASK_VAL;
    inverse_round_function(c0, c1, sk, x, y);
    uint32_t res = (x << 16) ^ y;
    return(res);
}

inline uint32_t encrypt_one_round(uint32_t p, uint32_t sk){
    uint32_t x,y;
    uint32_t p0 = p >> 16; uint32_t p1 = p & MASK_VAL;
    round_function(p0,p1,sk,x,y);
    uint32_t res = (x << 16) ^ y;
    return(res);
}

uint32_t encrypt(uint32_t p, uint64_t key, int rounds){
    uint32_t a = p >> WORD_SIZE; uint32_t b = p & MASK_VAL;
    uint16_t k[4];
    memcpy(k, &key, 8);
    uint32_t sk = k[3]; uint32_t tmp;
    for (uint32_t i = 0; i < rounds; i++){
        round_function(a,b,sk,a,b);
        round_function(k[2 - i%3], sk, i, tmp, sk);
        k[2-i%3] = tmp;
    }
    uint32_t res = (a << WORD_SIZE) + b;
    return(res);
}

uint32_t decrypt(uint32_t p, uint64_t key, int rounds){
    uint32_t a = p >> WORD_SIZE; uint32_t b = p & MASK_VAL;
    uint16_t k[4];
    memcpy(k, &key, 8);
    uint32_t sk = k[3]; uint32_t tmp;
    uint32_t ks[MAX_ROUNDS];
    for (uint32_t i = 0; i < rounds; i++){
        ks[i] = sk;
        round_function(k[2 - i%3], sk, i, tmp, sk);
        k[2-i%3] = tmp;
    }
    for (int i = rounds-1; i >= 0; i--){
        inverse_round_function(a,b,ks[i],a,b);
    }
    uint32_t res = (a << WORD_SIZE) + b;
    return(res);
}

void make_examples(uint32_t nr, uint32_t diff, vector<uint32_t>& v0, vector<uint32_t>& v1, vector<uint32_t>& w){
    random_device rd;
    uniform_int_distribution<uint32_t> rng32(0, 0xffffffff);
    uniform_int_distribution<uint64_t> rng64(0, 0xffffffffffffffffL);
    mt19937 rng(rng64(rd));
    for (int i = 0; i < w.size(); i++)
        w[i] = (rng32(rd)) & 1;
    for (int i = 0; i < v0.size(); i++){
        if (w[i]) {
            uint64_t key = rng64(rd);
            uint32_t plain0 = rng32(rd);
            uint32_t plain1 = plain0 ^ diff;
            uint32_t c0 = encrypt(plain0, key, nr);
            uint32_t c1 = encrypt(plain1, key, nr);
            v0[i] = c0; v1[i] = c1;
        } else {
            v0[i] = rng32(rd); v1[i] = rng32(rd);
            while (v0[i] == v1[i])
                v0[i] = rng32(rd);
        }
    }
}

double zmin = 1e-32;

inline uint32_t GET1(const uint32_t a, const uint32_t i)
{
    return ((a & (1U << i)) >> i);
}

inline void replace_1bit(uint32_t & a, const uint32_t b, const uint32_t i)
{
    a = a & (MASK_VAL ^ (1U << i));
    a = a | (b << i);
}

inline bool eq(const uint32_t a, const uint32_t b, const uint32_t c)
{
    return (1U == ((1U ^ a ^ b) & (1U ^ a ^ c)));
}

inline bool case_xy(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i)
{
    return 1U == ((1U ^ eq(alpha_i, beta_i, gamma_i)) & (alpha_i ^ beta_i ^ 1U));
}
inline bool case_xy_0(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == ((1U ^ eq(alpha_i, beta_i, gamma_i)) & (alpha_i ^ beta_i ^ 1U) & (alpha_i_p ^ beta_i_p ^ gamma_i_p ^ alpha_i ^ 1U));
}
inline bool case_xy_1(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == ((1U ^ eq(alpha_i, beta_i, gamma_i)) & (alpha_i ^ beta_i ^ 1U) & (alpha_i_p ^ beta_i_p ^ gamma_i_p ^ alpha_i ^ 0U));
}

inline bool case_xc(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i)
{
    return 1U == ((1U ^ eq(alpha_i, beta_i, gamma_i)) & ((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 1U));
}
inline bool case_xc_0(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == ((1U ^ eq(alpha_i, beta_i, gamma_i)) & ((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 1U) & ((alpha_i_p ^ beta_i_p ^ gamma_i_p ^ alpha_i) ^ 1U));
}
inline bool case_xc_1(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == ((1U ^ eq(alpha_i, beta_i, gamma_i)) & ((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 1U) & ((alpha_i_p ^ beta_i_p ^ gamma_i_p ^ alpha_i) ^ 0U));
}

inline bool case_yc(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i)
{
    return 1U == ((1U ^ eq(alpha_i, beta_i, gamma_i)) & ((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 0U));
}
inline bool case_yc_0(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == ((1U ^ eq(alpha_i, beta_i, gamma_i)) & ((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 0U) & ((alpha_i_p ^ beta_i_p ^ gamma_i_p ^ beta_i) ^ 1U));
}
inline bool case_yc_1(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == ((1U ^ eq(alpha_i, beta_i, gamma_i)) & ((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 0U) & ((alpha_i_p ^ beta_i_p ^ gamma_i_p ^ beta_i) ^ 0U));
}


inline bool case_xy_neq(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i)
{
    return 1U == (alpha_i ^ beta_i ^ 1U);
}
inline bool case_xy_0_neq(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == ((alpha_i ^ beta_i ^ 1U) & (alpha_i_p ^ beta_i_p ^ gamma_i_p ^ alpha_i ^ 1U));
}
inline bool case_xy_1_neq(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == ((alpha_i ^ beta_i ^ 1U) & (alpha_i_p ^ beta_i_p ^ gamma_i_p ^ alpha_i ^ 0U));
}

inline bool case_xc_neq(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i)
{
    return 1U == (((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 1U));
}
inline bool case_xc_0_neq(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == (((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 1U) & ((alpha_i_p ^ beta_i_p ^ gamma_i_p ^ alpha_i) ^ 1U));
}
inline bool case_xc_1_neq(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == (((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 1U) & ((alpha_i_p ^ beta_i_p ^ gamma_i_p ^ alpha_i) ^ 0U));
}

inline bool case_yc_neq(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i)
{
    return 1U == (((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 0U));
}
inline bool case_yc_0_neq(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == (((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 0U) & ((alpha_i_p ^ beta_i_p ^ gamma_i_p ^ beta_i) ^ 1U));
}
inline bool case_yc_1_neq(const uint32_t alpha_i, const uint32_t beta_i, const uint32_t gamma_i, const uint32_t alpha_i_p, const uint32_t beta_i_p, const uint32_t gamma_i_p)
{
    return 1U == (((alpha_i ^ beta_i) ^ 0U) & ((beta_i ^ gamma_i) ^ 0U) & ((alpha_i_p ^ beta_i_p ^ gamma_i_p ^ beta_i) ^ 0U));
}

#define B 6
#define mask (((1U << (B)) - 1U))
#define maskm (((1U << ((B) - 1U)) - 1U))


double A_pr[1 << B][1 << B][1 << B][1 << B];
double A_c_pr[1 << B][1 << B][1 << B][1 << B][2];
double A0_pr[1 << B][1 << B][1 << B][1 << B];
double A_next_pr[1 << B][1 << B][1 << B][1 << (B-1)][2];
double A_c_next_pr[1 << B][1 << B][1 << B][1 << (B-1)][2][2];

void initAs()
{
    for (uint32_t beta_b = 0; beta_b < (1 << B); beta_b++)
    {
        for (uint32_t gamma_b = 0; gamma_b < (1 << B); gamma_b++)
        {
            for (uint32_t y_b = 0; y_b < (1 << B); y_b++)
            {
                for (uint32_t alpha_b = 0; alpha_b < (1 << B); alpha_b++)
                {
                    A_pr[beta_b][gamma_b][y_b][alpha_b] = 0.0;
                    A_c_pr[beta_b][gamma_b][y_b][alpha_b][0] = 0.0;
                    A_c_pr[beta_b][gamma_b][y_b][alpha_b][1] = 0.0;
                    A0_pr[beta_b][gamma_b][y_b][alpha_b] = 0.0;
                }
                for (uint32_t alpha_b = 0; alpha_b < (1 << (B - 1)); alpha_b++)
                {
                    A_next_pr[beta_b][gamma_b][y_b][alpha_b][0U] = 0.0;
                    A_c_next_pr[beta_b][gamma_b][y_b][alpha_b][0][0U] = 0.0;
                    A_c_next_pr[beta_b][gamma_b][y_b][alpha_b][1][0U] = 0.0;
                    A_next_pr[beta_b][gamma_b][y_b][alpha_b][1U] = 0.0;
                    A_c_next_pr[beta_b][gamma_b][y_b][alpha_b][0][1U] = 0.0;
                    A_c_next_pr[beta_b][gamma_b][y_b][alpha_b][1][1U] = 0.0;
                }
            }
        }
    }
}

void Gen_Multi_Bits_Filter_Pr()
{
    auto start = std::chrono::high_resolution_clock::now();
    initAs();
    for (uint32_t beta = 0; beta < (1 << B); beta++)
    {
        for (uint32_t y0 = 0; y0 < (1 << B); y0++)
        {
            for (uint32_t alpha = 0; alpha < (1 << B); alpha++)
            {
                for (uint32_t x0 = 0; x0 < (1 << B); x0++)
                {
                    uint32_t x1 = x0 ^ alpha;
                    uint32_t y1 = y0 ^ beta;
                    uint32_t z0 = (x0 + y0) & mask;
                    uint32_t z1 = (x1 + y1) & mask;
                    uint32_t gamma = z0 ^ z1;
                    A0_pr[beta][gamma][y0][alpha] += 1.0;
                    for (uint32_t c0 = 0; c0 < (1 << 1); c0++)
                    {
                        for (uint32_t c1 = 0; c1 < (1 << 1); c1++)
                        {
                            uint32_t z0 = (x0 + y0 + c0) & mask;
                            uint32_t z1 = (x1 + y1 + c1) & mask;
                            uint32_t gamma = z0 ^ z1;
                            A_pr[beta][gamma][y0][alpha] += 1.0;
                            A_c_pr[beta][gamma][y0][alpha][c0] += 1.0;
                        }
                    }
                }
                for (uint32_t gamma = 0; gamma < (1 << B); gamma++)
                {
                    A0_pr[beta][gamma][y0][alpha] = A0_pr[beta][gamma][y0][alpha] / (double)(1 << B);
                    A_pr[beta][gamma][y0][alpha] = A_pr[beta][gamma][y0][alpha] / (double)(1 << (B + 2));
                    A_c_pr[beta][gamma][y0][alpha][0] = A_c_pr[beta][gamma][y0][alpha][0] / (double)(1 << (B + 1));
                    A_c_pr[beta][gamma][y0][alpha][1] = A_c_pr[beta][gamma][y0][alpha][1] / (double)(1 << (B + 1));
                }
            }
        }
    }
    for (uint32_t beta = 0; beta < (1 << B); beta++)
    {
        for (uint32_t gamma = 0; gamma < (1 << B); gamma++)
        {
            for (uint32_t y0 = 0; y0 < (1 << B); y0++)
            {
                for (uint32_t alpha = 0; alpha < (1 << (B - 1)); alpha++)
                {
                    if ((A_pr[beta][gamma][y0][(0<<(B-1)) | alpha] + A_pr[beta][gamma][y0][(1<<(B-1)) | alpha]) == 0.0)
                    {
                        A_next_pr[beta][gamma][y0][alpha][0] = 0.0;
                        A_next_pr[beta][gamma][y0][alpha][1] = 0.0;
                    }
                    else
                    {
                        A_next_pr[beta][gamma][y0][alpha][0] = 
                             A_pr[beta][gamma][y0][(0<<(B-1)) | alpha] / 
                            (A_pr[beta][gamma][y0][(0<<(B-1)) | alpha] + A_pr[beta][gamma][y0][(1<<(B-1)) | alpha]);
                        A_next_pr[beta][gamma][y0][alpha][1] = 
                             A_pr[beta][gamma][y0][(1<<(B-1)) | alpha] / 
                            (A_pr[beta][gamma][y0][(0<<(B-1)) | alpha] + A_pr[beta][gamma][y0][(1<<(B-1)) | alpha]);
                    }
                    if ((A_c_pr[beta][gamma][y0][(0<<(B-1)) | alpha][0] + A_c_pr[beta][gamma][y0][(1<<(B-1)) | alpha][0]) == 0.0)
                    {
                        A_c_next_pr[beta][gamma][y0][alpha][0][0] = 0.0;
                        A_c_next_pr[beta][gamma][y0][alpha][0][1] = 0.0;
                    }
                    else
                    {
                        A_c_next_pr[beta][gamma][y0][alpha][0][0] = 
                             A_c_pr[beta][gamma][y0][(0<<(B-1)) | alpha][0] / 
                            (A_c_pr[beta][gamma][y0][(0<<(B-1)) | alpha][0] + A_c_pr[beta][gamma][y0][(1<<(B-1)) | alpha][0]);
                        A_c_next_pr[beta][gamma][y0][alpha][0][1] = 
                             A_c_pr[beta][gamma][y0][(1<<(B-1)) | alpha][0] / 
                            (A_c_pr[beta][gamma][y0][(0<<(B-1)) | alpha][0] + A_c_pr[beta][gamma][y0][(1<<(B-1)) | alpha][0]);
                    }
                    if ((A_c_pr[beta][gamma][y0][(0<<(B-1)) | alpha][1] + A_c_pr[beta][gamma][y0][(1<<(B-1)) | alpha][1]) == 0.0)
                    {
                        A_c_next_pr[beta][gamma][y0][alpha][1][0] = 0.0;
                        A_c_next_pr[beta][gamma][y0][alpha][1][1] = 0.0;
                    }
                    else
                    {
                        A_c_next_pr[beta][gamma][y0][alpha][1][0] = 
                             A_c_pr[beta][gamma][y0][(0<<(B-1)) | alpha][1] / 
                            (A_c_pr[beta][gamma][y0][(0<<(B-1)) | alpha][1] + A_c_pr[beta][gamma][y0][(1<<(B-1)) | alpha][1]);
                        A_c_next_pr[beta][gamma][y0][alpha][1][1] = 
                             A_c_pr[beta][gamma][y0][(1<<(B-1)) | alpha][1] / 
                            (A_c_pr[beta][gamma][y0][(0<<(B-1)) | alpha][1] + A_c_pr[beta][gamma][y0][(1<<(B-1)) | alpha][1]);
                    }
                }
            }
        }
    }
    ofstream fout("./A0_pr" + to_string(B) + ".bin", ios::out | ios::binary);
    if (fout.is_open())
    {
        fout.write(reinterpret_cast<const char*>(&A0_pr[0]), sizeof(A0_pr));
        fout.close();
    } else
    {
        std::cerr << "Failed to open file for writing." << std::endl;
    }
    fout.open("./A_next_pr" + to_string(B) + ".bin", ios::out | ios::binary);
    if (fout.is_open())
    {
        fout.write(reinterpret_cast<const char*>(&A_next_pr[0]), sizeof(A_next_pr));
        fout.close();
    } else
    {
        std::cerr << "Failed to open file for writing." << std::endl;
    }
    fout.open("./A_c_next_pr" + to_string(B) + ".bin", ios::out | ios::binary);
    if (fout.is_open())
    {
        fout.write(reinterpret_cast<const char*>(&A_c_next_pr[0]), sizeof(A_c_next_pr));
        fout.close();
    } else
    {
        std::cerr << "Failed to open file for writing." << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Gen_Multi_Bits_Filter_Pr() execution time: " << duration.count() << " seconds" << std::endl;
    #if 0
    fout.open("./A0_pr" + to_string(B) + ".txt", ios::out);
    for (uint32_t beta = 0; beta < (1 << B); beta++)
    {
        fout << "beta = 0x" << hex << beta << endl;
        for (uint32_t gamma = 0; gamma < (1 << B); gamma++)
        {
            fout << "gamma = 0x" << hex << gamma << endl;
            for (uint32_t y0 = 0; y0 < (1 << B); y0++)
            {
                fout << "y0 = 0x" << hex << y0 << endl;
                for (uint32_t alpha = 0; alpha < (1 << B); alpha++)
                {
                    fout << A0_pr[beta][gamma][y0][alpha] << ", ";
                }
                fout << endl;
            }
        }
    }
    fout.close();
    fout.open("./A_next_pr" + to_string(B) + ".txt", ios::out);
    for (uint32_t beta = 0; beta < (1 << B); beta++)
    {
        fout << "beta = 0x" << hex << beta << endl;
        for (uint32_t gamma = 0; gamma < (1 << B); gamma++)
        {
            fout << "gamma = 0x" << hex << gamma << endl;
            for (uint32_t y0 = 0; y0 < (1 << B); y0++)
            {
                fout << "y0 = 0x" << hex << y0 << endl;
                for (uint32_t alpha = 0; alpha < (1 << (B - 1)); alpha++)
                {
                    fout << "alpha = 0x" << hex << alpha << endl;
                    for (uint32_t alpha_0 = 0; alpha_0 < (1 << 1); alpha_0++)
                    {
                        fout << A_next_pr[beta][gamma][y0][alpha][alpha_0] << ", ";
                    }
                    fout << endl;
                }
            }
        }
    }
    fout.close();
    fout.open("./A_c_next_pr" + to_string(B) + ".txt", ios::out);
    for (uint32_t beta = 0; beta < (1 << B); beta++)
    {
        fout << "beta = 0x" << hex << beta << endl;
        for (uint32_t gamma = 0; gamma < (1 << B); gamma++)
        {
            fout << "gamma = 0x" << hex << gamma << endl;
            for (uint32_t y0 = 0; y0 < (1 << B); y0++)
            {
                fout << "y0 = 0x" << hex << y0 << endl;
                for (uint32_t alpha = 0; alpha < (1 << (B - 1)); alpha++)
                {
                    fout << "alpha = 0x" << hex << alpha << endl;
                    for (uint32_t c0 = 0; c0 < (1 << 1); c0++)
                    {
                        fout << "c0 = 0x" << hex << c0 << endl;
                        for (uint32_t alpha_0 = 0; alpha_0 < (1 << 1); alpha_0++)
                        {
                            fout << A_c_next_pr[beta][gamma][y0][alpha][c0][alpha_0] << ", ";
                        }
                        fout << endl;
                    }
                }
            }
        }
    }
    fout.close();
    #endif
}

void Load_Multi_Bits_Filter_Pr(const std::string& prefixA_c_next_pr, const std::string& prefixA_next_pr, const std::string& prefixA0_pr)
{
    ifstream fin(prefixA0_pr, ios::in | ios::binary);
    if (fin.is_open())
    {
        fin.read(reinterpret_cast<char*>(&A0_pr[0]), sizeof(A0_pr));
        fin.close();
    } else
    {
        std::cerr << "Failed to open file for writing." << std::endl;
    }
    fin.open(prefixA_next_pr, ios::in | ios::binary);
    if (fin.is_open())
    {
        fin.read(reinterpret_cast<char*>(&A_next_pr[0]), sizeof(A_next_pr));
        fin.close();
    } else
    {
        std::cerr << "Failed to open file for writing." << std::endl;
    }
    fin.open(prefixA_c_next_pr, ios::in | ios::binary);
    if (fin.is_open())
    {
        fin.read(reinterpret_cast<char*>(&A_c_next_pr[0]), sizeof(A_c_next_pr));
        fin.close();
    } else
    {
        std::cerr << "Failed to open file for writing." << std::endl;
    }
}

bool fileExists(const std::string& filename)
{
    std::ifstream file(filename.c_str());
    return file.good();
}

double avg_num_alpha = 0.0;

void ComputeAlphaPr_nextbit(
    std::array<uint32_t, WORD_SIZE> & c_bits, 
    std::array<uint32_t, WORD_SIZE> & alpha_bits, 
    std::array<uint32_t, WORD_SIZE> & beta_bits, 
    std::array<uint32_t, WORD_SIZE> & gamma_bits, 
    std::array<uint32_t, WORD_SIZE> & y_bits, 
    uint32_t bi, 
    double pr, 
    double & sumPr, 
    vector<double> & net_DD, 
    uint32_t & diff, 
    uint32_t & beta, 
    uint32_t & gamma, 
    uint32_t & y, 
    uint32_t & alpha)
{
    double cur_pr = 0.0;
    double c_bits_next_org = 0.0;
    if (sumPr > r0) return;
    if (bi == (WORD_SIZE - 1))
    {
        //avg_num_alpha = avg_num_alpha + 1.0;
        uint32_t cur_diff = diff;
        for (uint32_t bj = 0; bj < WORD_SIZE; bj++)
        {
            cur_diff |= (alpha_bits[bj] << ((bj + ALPHA) & 0xf));
        }
        sumPr = sumPr + pr * net_DD[cur_diff];
        return;
    }
    // try to determine c_bits[bi + 1]
    // c_bits[bi + 1] = x_bits[bi] & y_bits[bi] ^ (x_bits[bi] ^ y_bits[bi]) & c_bits[bi]
    if ((y_bits[bi] == 0) && (c_bits[bi] == 0)) 
    {
        c_bits[bi + 1] = 0;
    } else if ((y_bits[bi] == 1) && (c_bits[bi] == 1))
    {
        c_bits[bi + 1] = 1;
    } else
    {
        c_bits[bi + 1] = 2; // 2 means unknown
    }
    if (eq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi]))
    {
        alpha_bits[bi + 1] = beta_bits[bi + 1] ^ gamma_bits[bi + 1] ^ beta_bits[bi];
        replace_1bit(alpha, alpha_bits[bi + 1], bi + 1);
        ComputeAlphaPr_nextbit(c_bits, alpha_bits, beta_bits, gamma_bits, y_bits, bi + 1, pr, sumPr, net_DD, diff, beta, gamma, y, alpha);
        return;
    } else if (case_yc_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi]))
    {
        if (c_bits[bi] != 2) // known c_bits[bi]
        {
            alpha_bits[bi + 1] = beta_bits[bi + 1] ^ gamma_bits[bi + 1] ^ beta_bits[bi] ^ y_bits[bi] ^ c_bits[bi];
            if (case_yc_0_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]))
            {
                c_bits[bi + 1] = y_bits[bi];
            }
            replace_1bit(alpha, alpha_bits[bi + 1], bi + 1);
            ComputeAlphaPr_nextbit(c_bits, alpha_bits, beta_bits, gamma_bits, y_bits, bi + 1, pr, sumPr, net_DD, diff, beta, gamma, y, alpha);
            return;
        }
        else // unknown c_bits[bi]
        {
            // alpha_bits[bi + 1] can be 0 and 1
            uint32_t alpha_b = (alpha >> (bi - B + 2)) & maskm;
            uint32_t beta_b = (beta >> (bi - B + 2)) & mask;
            uint32_t gamma_b = (gamma >> (bi - B + 2)) & mask;
            uint32_t y_b = (y >> (bi - B + 2)) & mask;
            if (c_bits[bi - B + 2] != 2)
            {
                cur_pr = A_c_next_pr[beta_b][gamma_b][y_b][alpha_b][c_bits[bi - B + 2]][0];
            }
            else
            {
                cur_pr = A_next_pr[beta_b][gamma_b][y_b][alpha_b][0];
            }
            if (cur_pr > 0.0)
            {
                alpha_bits[bi + 1] = 0;
                c_bits_next_org = c_bits[bi + 1];
                if (case_yc_0_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]))
                {
                    c_bits[bi + 1] = y_bits[bi];
                }
                replace_1bit(alpha, alpha_bits[bi + 1], bi + 1);
                ComputeAlphaPr_nextbit(c_bits, alpha_bits, beta_bits, gamma_bits, y_bits, bi + 1, pr * cur_pr, sumPr, net_DD, diff, beta, gamma, y, alpha);
                c_bits[bi + 1] = c_bits_next_org;
            }

            if (c_bits[bi - B + 2] != 2)
            {
                cur_pr = A_c_next_pr[beta_b][gamma_b][y_b][alpha_b][c_bits[bi - B + 2]][1];
            }
            else
            {
                cur_pr = A_next_pr[beta_b][gamma_b][y_b][alpha_b][1];
            }
            if (cur_pr > 0.0)
            {
                alpha_bits[bi + 1] = 1;
                if (case_yc_0_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]))
                {
                    c_bits[bi + 1] = y_bits[bi];
                }
                replace_1bit(alpha, alpha_bits[bi + 1], bi + 1);
                ComputeAlphaPr_nextbit(c_bits, alpha_bits, beta_bits, gamma_bits, y_bits, bi + 1, pr * cur_pr, sumPr, net_DD, diff, beta, gamma, y, alpha);
            }
            return;
        }
    }
    else
    {
        // alpha_bits[bi + 1] can be 0 and 1
        uint32_t alpha_b = (alpha >> (bi - B + 2)) & maskm;
        uint32_t beta_b = (beta >> (bi - B + 2)) & mask;
        uint32_t gamma_b = (gamma >> (bi - B + 2)) & mask;
        uint32_t y_b = (y >> (bi - B + 2)) & mask;
        if (c_bits[bi - B + 2] != 2)
        {
            cur_pr = A_c_next_pr[beta_b][gamma_b][y_b][alpha_b][c_bits[bi - B + 2]][0];
        }
        else
        {
            cur_pr = A_next_pr[beta_b][gamma_b][y_b][alpha_b][0];
        }
        if (cur_pr > 0.0)
        {
            alpha_bits[bi + 1] = 0;
            c_bits_next_org = c_bits[bi + 1];
            if (case_xy_0_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]) ||
                case_xc_1_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]))
            {
                c_bits[bi + 1] = y_bits[bi];
            }
            else if (case_xy_1_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]) ||
                     case_xc_0_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]))
            {
                c_bits[bi + 1] = c_bits[bi];
            }
            replace_1bit(alpha, alpha_bits[bi + 1], bi + 1);
            ComputeAlphaPr_nextbit(c_bits, alpha_bits, beta_bits, gamma_bits, y_bits, bi + 1, pr * cur_pr, sumPr, net_DD, diff, beta, gamma, y, alpha);
            c_bits[bi + 1] = c_bits_next_org;
        }
        if (c_bits[bi - B + 2] != 2)
        {
            cur_pr = A_c_next_pr[beta_b][gamma_b][y_b][alpha_b][c_bits[bi - B + 2]][1];
        }
        else
        {
            cur_pr = A_next_pr[beta_b][gamma_b][y_b][alpha_b][1];
        }
        if (cur_pr > 0.0)
        {
            alpha_bits[bi + 1] = 1;
            if (case_xy_0_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]) ||
                case_xc_1_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]))
            {
                c_bits[bi + 1] = y_bits[bi];
            }
            else if (case_xy_1_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]) ||
                     case_xc_0_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]))
            {
                c_bits[bi + 1] = c_bits[bi];
            }
            replace_1bit(alpha, alpha_bits[bi + 1], bi + 1);
            ComputeAlphaPr_nextbit(c_bits, alpha_bits, beta_bits, gamma_bits, y_bits, bi + 1, pr * cur_pr, sumPr, net_DD, diff, beta, gamma, y, alpha);
        }
        return;
    }
}

double Compute_ConditionalPr(vector<double> & net_DD, uint32_t & beta, uint32_t & gamma, uint32_t & y)
{
    std::array<uint32_t, WORD_SIZE> beta_bits;
    std::array<uint32_t, WORD_SIZE> gamma_bits;
    std::array<uint32_t, WORD_SIZE> y_bits;
    std::array<uint32_t, WORD_SIZE> alpha_bits;
    std::array<uint32_t, WORD_SIZE> c_bits;
    for (size_t bi = 0; bi < WORD_SIZE; ++bi)
    {
        beta_bits[bi] = GET1(beta, bi);
        gamma_bits[bi] = GET1(gamma, bi);
        y_bits[bi] = GET1(y, bi);
    }
    alpha_bits.fill(0U); // Initialized to 0s
    c_bits.fill(0U);     // Initialized to 0s
    uint32_t alpha = 0U;
    uint32_t diff = beta << WORD_SIZE;
    double sumPr = 0.0;
    double pr = 1.0;
    double cur_pr = 0.0;
    for (uint32_t alpha_b = 0; alpha_b < (1 << B); alpha_b++)
    {
        cur_pr = A0_pr[beta & mask][gamma & mask][y & mask][alpha_b];
        if (cur_pr > 0.0)
        {
            alpha = alpha_b;
            for (size_t bi = 0; bi < B; ++bi)
            {
                alpha_bits[bi] = GET1(alpha, bi);
            }
            for (int bi = 0; bi < B - 1; ++bi)
            {
                if ((y_bits[bi] == 0) && (c_bits[bi] == 0)) {
                    c_bits[bi + 1] = 0;
                } 
                else if ((y_bits[bi] == 1) && (c_bits[bi] == 1)) {
                    c_bits[bi + 1] = 1;
                } 
                else if (!eq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi])) {
                    if (case_xy_0_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]) ||
                        case_xc_1_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]) ||
                        case_yc_0_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1])) {
                        c_bits[bi + 1] = y_bits[bi];
                    } 
                    else if (case_xy_1_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1]) ||
                             case_xc_0_neq(alpha_bits[bi], beta_bits[bi], gamma_bits[bi], alpha_bits[bi + 1], beta_bits[bi + 1], gamma_bits[bi + 1])) {
                        c_bits[bi + 1] = c_bits[bi];
                    } 
                    else {
                        c_bits[bi + 1] = 2; // 2 means unknown
                    }
                } 
                else {
                    c_bits[bi + 1] = 2; // 2 means unknown
                }
            }
            ComputeAlphaPr_nextbit(c_bits, alpha_bits, beta_bits, gamma_bits, y_bits, B - 1, pr * cur_pr, sumPr, net_DD, diff, beta, gamma, y, alpha);
        }
    }
    return sumPr;
}

void eval_conditional_DDT(int TDN, vector<uint32_t> & Z_DD, vector<uint32_t> & X0, vector<uint32_t> & X1, vector<double> & net_DD, int num_rounds)
{
    for (size_t i = 0; i < TDN; i++)
    {
        uint32_t C0_L = (X0[i] >> WORD_SIZE) & MASK_VAL;
        uint32_t C0_R = X0[i] & MASK_VAL;
        uint32_t dC = X0[i] ^ X1[i];
        uint32_t dx = (dC >> WORD_SIZE) & MASK_VAL;
        uint32_t dy = dC & MASK_VAL;
        uint32_t beta = ror(dx ^ dy, BETA);
        uint32_t gamma = dx;
        uint32_t y = ror(C0_L ^ C0_R, BETA);
        double pr = Compute_ConditionalPr(net_DD, beta, gamma, y);
        if (pr > r0)
        {
            Z_DD[i] = 1;
        } else
        {
            Z_DD[i] = 0;
        }
    }
}

void EvaluateConditionalDDT(int num_rounds)
{
    int TDN = 1 << TEST_DATA_logN;
    uint32_t input_diff = 0x00400000U;
    string wdir = "./";
    ofstream logfile(wdir + "5R_8R_speck_eval_conditional_DDT_withPr_carry_TND" + to_string(int(log2(TDN))) + ".log", ios::out | ios::app);
    logfile << "==== " << num_rounds << " rounds" << endl;

    uint64_t num_diffs = 1L << 32;
    vector<double> net_DD(num_diffs);
    string trained_model_DD = wdir + "../../ddt_400000_" + to_string(num_rounds - 1) + "rounds.bin";
    ifstream fin(trained_model_DD, ios::in | ios::binary);
    fin.read((char*)&net_DD[0], net_DD.size() * sizeof(double));
    logfile << trained_model_DD << endl;
    fin.close();

    std::string prefixA_c_next_pr = "./A_c_next_pr" + std::to_string(B) + ".bin";
    std::string prefixA_next_pr = "./A_next_pr" + std::to_string(B) + ".bin";
    std::string prefixA0_pr = "./A0_pr" + std::to_string(B) + ".bin";
    if (fileExists(prefixA_c_next_pr) && fileExists(prefixA_next_pr) && fileExists(prefixA0_pr))
    {
        Load_Multi_Bits_Filter_Pr(prefixA_c_next_pr, prefixA_next_pr, prefixA0_pr);
    } else
    {
        Gen_Multi_Bits_Filter_Pr();
    }
    for (int ti = 0; ti < 1; ti++)
    {
        //avg_num_alpha = 0.0;
        logfile << "Test " << ti << endl;
        vector<uint32_t> X0(TDN);
        vector<uint32_t> X1(TDN);
        vector<uint32_t> Y(TDN);
        vector<uint32_t> Z_DD(TDN);
        vector<double> diff(TDN);

        make_examples(num_rounds, input_diff, X0, X1, Y);
        std::clock_t start_clock = std::clock();
        eval_conditional_DDT(TDN, Z_DD, X0, X1, net_DD, num_rounds);
        double duration_clock = (std::clock() - start_clock) / static_cast<double>(CLOCKS_PER_SEC);

        for (size_t i = 0; i < TDN; ++i)
        {
            diff[i] = (double)Y[i] - (double)Z_DD[i];
        }
        // mses = np.mean(diff*diff);
        double mses_sum = 0.0;
        for (size_t i = 0; i < TDN; ++i)
        {
            mses_sum += diff[i] * diff[i];
        }
        double mses = mses_sum / (double)TDN;
        // n = len(Z_DD);
        size_t n = TDN;
        // n0 = np.sum(Y==0); n1 = np.sum(Y==1);
        size_t n0 = 0, n1 = 0;
        for (size_t i = 0; i < TDN; ++i)
        {
            if (Y[i] == 0) n0++;
            if (Y[i] == 1) n1++;
        }
        // accs = np.sum(Z_DD == Y) / n;
        size_t acc_count = 0;
        for (size_t i = 0; i < TDN; ++i)
        {
            if (Z_DD[i] == Y[i]) acc_count++;
        }
        double accs = static_cast<double>(acc_count) / n;
        // tprs = np.sum(Z_DD[Y==1]) / n1;
        double tprs_sum = 0.0;
        for (size_t i = 0; i < TDN; ++i)
        {
            if (Y[i] == 1) tprs_sum += Z_DD[i];
        }
        double tprs = tprs_sum / n1;
        // tnrs = np.sum(Z_DD[Y==0] == 0) / n0;
        size_t tnrs_count = 0;
        for (size_t i = 0; i < TDN; ++i)
        {
            if ((Y[i] == 0) && (Z_DD[i] == 0)) tnrs_count++;
        }
        double tnrs = static_cast<double>(tnrs_count) / n0;
        //logfile << "avg_num_alpha: 2^" << log2(avg_num_alpha/TDN) << endl;
        logfile << "ACC " << accs << "    TPR " << tprs << "    TNR " << tnrs << "    MSE " << mses << endl;
        logfile << "Time taken:  " << duration_clock << " seconds" << " = 2^" << log2(duration_clock) << " seconds" << std::endl;
        logfile << std::endl;
    }
    logfile.close();
}

void eval_DDT(int TDN, vector<uint32_t> & Z_DD, vector<uint32_t> & X0, vector<uint32_t> & X1, vector<double> & net_DD, int num_rounds)
{
    for (size_t i = 0; i < TDN; i++)
    {
        uint32_t dC = X0[i] ^ X1[i];
        dC = ((dC >> WORD_SIZE) & MASK_VAL) | ((dC & MASK_VAL) << WORD_SIZE);
        double pr = net_DD[dC];
        if (pr > r0)
        {
            Z_DD[i] = 1;
        } else
        {
            Z_DD[i] = 0;
        }
    }
}

void EvaluateDDT(int num_rounds)
{
    int TDN = 1 << TEST_DATA_logN;
    uint32_t input_diff = 0x00400000U;
    string wdir = "./";
    ofstream logfile(wdir + "5R_8R_speck_eval_DDT_TND" + to_string(int(log2(TDN))) + ".log", ios::out | ios::app);
    logfile << "==== " << num_rounds << " rounds" << endl;

    uint64_t num_diffs = 1L << 32;
    vector<double> net_DD(num_diffs);
    string trained_model_DD = wdir + "../../ddt_400000_" + to_string(num_rounds) + "rounds.bin";
    ifstream fin(trained_model_DD, ios::in | ios::binary);
    fin.read((char*)&net_DD[0], net_DD.size() * sizeof(double));
    logfile << trained_model_DD << endl;
    fin.close();

    for (int ti = 0; ti < 1; ti++)
    {
        logfile << "Test " << ti << endl;
        vector<uint32_t> X0(TDN);
        vector<uint32_t> X1(TDN);
        vector<uint32_t> Y(TDN);
        vector<uint32_t> Z_DD(TDN);
        vector<double> diff(TDN);

        make_examples(num_rounds, input_diff, X0, X1, Y);
        std::clock_t start_clock = std::clock();
        eval_DDT(TDN, Z_DD, X0, X1, net_DD, num_rounds);
        double duration_clock = (std::clock() - start_clock) / static_cast<double>(CLOCKS_PER_SEC);

        for (size_t i = 0; i < TDN; ++i)
        {
            diff[i] = (double)Y[i] - (double)Z_DD[i];
        }
        // mses = np.mean(diff*diff);
        double mses_sum = 0.0;
        for (size_t i = 0; i < TDN; ++i)
        {
            mses_sum += diff[i] * diff[i];
        }
        double mses = mses_sum / (double)TDN;
        // n = len(Z_DD);
        size_t n = TDN;
        // n0 = np.sum(Y==0); n1 = np.sum(Y==1);
        size_t n0 = 0, n1 = 0;
        for (size_t i = 0; i < TDN; ++i)
        {
            if (Y[i] == 0) n0++;
            if (Y[i] == 1) n1++;
        }
        // accs = np.sum(Z_DD == Y) / n;
        size_t acc_count = 0;
        for (size_t i = 0; i < TDN; ++i)
        {
            if (Z_DD[i] == Y[i]) acc_count++;
        }
        double accs = static_cast<double>(acc_count) / n;
        // tprs = np.sum(Z_DD[Y==1]) / n1;
        double tprs_sum = 0.0;
        for (size_t i = 0; i < TDN; ++i)
        {
            if (Y[i] == 1) tprs_sum += Z_DD[i];
        }
        double tprs = tprs_sum / n1;
        // tnrs = np.sum(Z_DD[Y==0] == 0) / n0;
        size_t tnrs_count = 0;
        for (size_t i = 0; i < TDN; ++i)
        {
            if ((Y[i] == 0) && (Z_DD[i] == 0)) tnrs_count++;
        }
        double tnrs = static_cast<double>(tnrs_count) / n0;
        logfile << "ACC " << accs << "    TPR " << tprs << "    TNR " << tnrs << "    MSE " << mses << endl;
        logfile << "Time taken:  " << duration_clock << " seconds" << " = 2^" << log2(duration_clock) << " seconds" << std::endl;
        logfile << std::endl;
    }
    logfile.close();
}

int main()
{
    for (int ri = 5; ri < 9; ri++)
    {
        EvaluateDDT(ri); // DD
    }
    for (int ri = 5; ri < 9; ri++)
    {
        EvaluateConditionalDDT(ri); // AD_YD
    }
    return 0;
}