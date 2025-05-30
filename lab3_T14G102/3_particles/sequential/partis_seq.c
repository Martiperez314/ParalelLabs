#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define RHO 1.2
#define CD 0.5
#define R 1e-5
#define M 1e-9
#define PI 3.14159265358979323846
#define A (PI * R * R)         // Front area of the particle
#define K (0.5 * RHO * CD * A) // drag coefficient

// Cone Angles limits
#define THETA_MIN 0.0
#define THETA_MAX (PI / 6.0)
#define PHI_MIN 0.0
#define PHI_MAX (2.0 * PI)

// Particle velocity limits
#define V_MIN 100.0
#define V_MAX 300.0

// Integration times
#define TOTAL_TIME 1e-3
#define DT 1e-6

// Saving solution
#define FREQ 100

typedef struct
{
    double x;
    double y;
    double z;
} Vector3D;

typedef struct
{
    Vector3D pos;
    Vector3D vel;
} Particle;

double random_double(const double min, const double max)
{
    return (rand() / (double)RAND_MAX) * (max - min) + min;
}

void setInitialConditions(Particle *particles, const int N)
{
    double theta, phi, v0;
    for (int i = 0; i < N; ++i)
    {
        // TODO
        // Initial position
        particles[i].pos.x = 0.0;
        particles[i].pos.y = 0.0;
        particles[i].pos.z = 0.0;

        // 2 particles for validation
        if (i == 0 || i == N - 1)
        {
            if (i == 0)
            {
                particles[0].vel.x = 2.830192e+00;
                particles[0].vel.y = 5.587800e+01;
                particles[0].vel.z = -2.226035e+02;
            }
            else
            {
                particles[N - 1].vel.x = 1.201285e+01;
                particles[N - 1].vel.y = 1.966662e+01;
                particles[N - 1].vel.z = -2.090078e+02;
            }
        }
        else
        {
            // TODO
            // Generate random velocity and direction
            v0 = random_double(V_MIN, V_MAX);
            theta = random_double(THETA_MIN, THETA_MAX);
            phi = random_double(PHI_MIN, PHI_MAX);

            particles[i].vel.x = v0 * sin(theta) * cos(phi);
            particles[i].vel.y = v0 * sin(theta) * sin(phi);
            particles[i].vel.z = -v0 * cos(theta);
        }
    }
}

// TODO
// Calculate new position and velocity
void integrateEuler(Particle *particles, const int N)
{
    double vx, vy, vz, vmag;
    for (int i = 0; i < N; ++i)
    {
        //current speed
        vx = particles[i].vel.x;
        vy = particles[i].vel.y;
        vz = particles[i].vel.z;

        vmag = sqrt(vx * vx + vy * vy + vz * vz);

        //new speed  (Newton 2. law)
        particles[i].vel.x -= (K / M) * vmag * vx * DT;
        particles[i].vel.y -= (K / M) * vmag * vy * DT;
        particles[i].vel.z -= (K / M) * vmag * vz * DT;

        //new pos (Newton 2. law)
        particles[i].pos.x += particles[i].vel.x * DT;
        particles[i].pos.y += particles[i].vel.y * DT;
        particles[i].pos.z += particles[i].vel.z * DT;
    }
}

// TODO
// Copy the state of the particles to a backup buffer.
void copyFrame(Particle *p_dst, Particle *p_src, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        p_dst[i].pos = p_src[i].pos;
        p_dst[i].vel = p_src[i].vel;
    }
}

int validate(Particle *particles, const int N)
{
    int flag = 0;
    double err = 1e-6;

    Particle p1 = particles[0];
    Particle p2 = particles[N - 1];
    // First particle
    if (fabs(p1.pos.x - 2.800044e-03) > err ||
        fabs(p1.pos.y - 5.528277e-02) > err ||
        fabs(p1.pos.z - (-2.202323e-01)) > err)
    {
        flag = 1;
        printf("Position of particle 0 is wrong: Expected: (%e,%e,%e)\tActual: (%e,%e,%e)\n",
               2.800044e-03, 5.528277e-02, -2.202323e-01, p1.pos.x, p1.pos.y, p1.pos.z);
    }
    // Second particle
    if (fabs(p2.pos.x - 1.189548e-02) > err ||
        fabs(p2.pos.y - 1.947447e-02) > err ||
        fabs(p2.pos.z - (-2.069657e-01)) > err)
    {
        flag = 1;
        printf("Position of particle (N - 1) is wrong: Expected: (%e,%e,%e)\tActual: (%e,%e,%e)\n",
               1.189548e-02, 1.947447e-02, -2.069657e-01, p2.pos.x, p2.pos.y, p2.pos.z);
    }
    return flag;
}

void write_solution(Particle *particles, const int N, const double t, char *filename)
{
    Particle *p = NULL;
    sprintf(filename, "out/time_%f.csv", t);

    FILE *f = fopen(filename, "w");
    for (int i = 0; i < N; ++i)
    {
        p = &particles[i];
        fprintf(f, "%e,%e,%e,%e,%e,%e\n",
                p->pos.x, p->pos.y, p->pos.z,
                p->vel.x, p->vel.y, p->vel.z);
    }
    fclose(f);
}

void read_solution(Particle *particles, const int N, char *filename)
{
    Particle *p = NULL;
    char line[1000];

    FILE *f = fopen(filename, "r");

    for (int i = 0; i < N; ++i)
    {
        p = &particles[i];

        if (fgets(line, 1000, f) == NULL)
        {
            printf("Error reading line %d of file %s\n", i, filename);
            return;
        }
        sscanf(line, "%le,%le,%le,%le,%le,%le\n",
               &(p->pos.x), &(p->pos.y), &(p->pos.z),
               &(p->vel.x), &(p->vel.y), &(p->vel.z));
    }
    fclose(f);
}

int main(int argc, char *argv[])
{
    int N;
    int write_flag;
    if (argc > 2)
    {
        N = atoi(argv[1]);
        write_flag = atoi(argv[2]);
    }
    else
    {
        printf("Usage: %s <particles N> <check>\n", argv[0]);
        return 1;
    }

    printf("Particles: %d\n", N);

    // Seed random generator
    srand(time(NULL));

    //
    // Allocate memory
    //
    Particle *particles = (Particle *)malloc(N * sizeof(Particle));
    Particle *pFrame = (Particle *)malloc(N * sizeof(Particle));

    //
    // Initial conditions
    //
    char filename[100];
    setInitialConditions(particles, N);

    // Create solution folder
    if (write_flag)
    {
        int system_flag = system("rm -rf ./out");
        system_flag = system("mkdir out");
        if (system_flag != 0) {
            printf("Error creating out directory\n");
            return 1;
        }
        write_solution(particles, N, 0.0, filename); // Write initial conditions
    }

    //
    // Particle simulation
    //
    int iter = 1;
    double t = 0.0;
    double curr_frame_time = 0.0;

    // Start timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    while (t <= TOTAL_TIME)
    {
        // Time integration
        integrateEuler(particles, N);
        t += DT;

        // Snapshot of the solution
        if (iter % FREQ == 0)
        {
            curr_frame_time = t;
            printf("Iter: %d. Saving snapshot t = %e in pFrame\n", iter, curr_frame_time);
            copyFrame(pFrame, particles, N);

            if (write_flag)
                write_solution(pFrame, N, curr_frame_time, filename);
        }

        ++iter;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

    //
    //  Validation
    //
    if (write_flag)
        read_solution(pFrame, N, filename);
    int return_flag = validate(pFrame, N);

    printf("Elapsed time: %f seconds\n", elapsed);

    free(particles);
    free(pFrame);

    return return_flag;
}