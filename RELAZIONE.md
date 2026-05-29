# Relazione Tecnica — Mini-Solver Jacobi per l'Equazione di Poisson 2D

> **Corso:** Parallel High Performance Computing — Prof. V. Mele  
> **Progetto 3** — Kernel pseudo-reale su singola GPU  
> **Autore relazione:** analisi automatica della codebase

---

## Indice

1. [Il Problema Matematico](#1-il-problema-matematico)
2. [Discretizzazione e Formula di Jacobi](#2-discretizzazione-e-formula-di-jacobi)
3. [Obiettivi Minimi della Traccia e Verifica del Rispetto](#3-obiettivi-minimi-della-traccia-e-verifica-del-rispetto)
4. [Architettura del Progetto — Mappa del Codice](#4-architettura-del-progetto--mappa-del-codice)
5. [Analisi File per File](#5-analisi-file-per-file)
   - [include/common.h](#51-includecommonh)
   - [include/timer.h](#52-includetimerh)
   - [include/cpu_solver.h](#53-includecpu_solverh)
   - [src/cpu_solver.cpp](#54-srccpu_solvercpp)
   - [include/reduction.h](#55-includereductionh)
   - [src/reduction.cu](#56-srcreductioncu)
   - [include/gpu_solver.h](#57-includegpu_solverh)
   - [src/gpu_solver.cu](#58-srcgpu_solvercu)
   - [include/validation.h](#59-includevalidationh)
   - [src/validation.cpp](#510-srcvalidationcpp)
   - [main.cu](#511-maincu)
6. [Linee Guida CUDA — Verifica Rispetto Best Practices](#6-linee-guida-cuda--verifica-rispetto-best-practices)
7. [Analisi delle Tre Varianti GPU](#7-analisi-delle-tre-varianti-gpu)
8. [Flusso di Esecuzione Completo](#8-flusso-di-esecuzione-completo)
9. [Punti Critici e Possibili Miglioramenti](#9-punti-critici-e-possibili-miglioramenti)

---

## 1. Il Problema Matematico

### 1.1 Equazione di Poisson 2D

Il progetto risolve l'**equazione di Poisson bidimensionale**:

$$-\Delta u(x,y) = f(x,y)$$

su un dominio rettangolare $\Omega = [0,1]^2$ con **condizioni di Dirichlet omogenee** sul bordo:

$$u = 0 \quad \text{su } \partial\Omega$$

Il Laplaciano $\Delta u$ è:

$$-\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = f$$

**Interpretazione fisica:** $f$ è un termine sorgente (per esempio una distribuzione di calore o una densità di carica); $u$ è la quantità incognita (temperatura, potenziale elettrico, etc.).

### 1.2 Scelta del termine sorgente e soluzione esatta

Il codice usa il termine sorgente:

$$f(x,y) = 2\pi^2 \sin(\pi x)\sin(\pi y)$$

Questo permette di avere una **soluzione analitica esatta**:

$$u_{\text{exact}}(x,y) = \sin(\pi x)\sin(\pi y)$$

**Verifica:** Applicando il Laplaciano:

$$-\Delta u_{\text{exact}} = -(-\pi^2 \sin(\pi x)\sin(\pi y) - \pi^2 \sin(\pi x)\sin(\pi y)) = 2\pi^2 \sin(\pi x)\sin(\pi y) = f$$

Questa scelta è fondamentale per la **fase di validazione**: permette di misurare quanto la soluzione numerica differisce dalla soluzione esatta, senza bisogno di riferimento esterno.

---

## 2. Discretizzazione e Formula di Jacobi

### 2.1 Griglia e spaziatura

La griglia è $N \times N$ con **spaziatura uniforme** (per discretizzare un dominio in intervalli di ampiezza costante):

$$h = \Delta x = \Delta y = \frac{1}{N-1}$$

L'elemento $(i,j)$ occupa la posizione fisica $(x, y) = (j \cdot h,\; i \cdot h)$.  
**Convenzione row-major:** `u[i,j]` è memorizzato all'indice `i*N + j`.  
Macro: `#define IDX(i,j,N) ((i)*(N)+(j))`

### 2.2 Differenze finite centrate

Approssimazione delle derivate seconde con **differenze finite centrate al secondo ordine** (errore $O(h^2)$):

$$\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2}$$

$$\frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2}$$

### 2.3 Stencil a 5 punti

Sostituendo nell'equazione e moltiplicando per $h^2$:

$$-u_{i+1,j} - u_{i-1,j} - u_{i,j+1} - u_{i,j-1} + 4u_{i,j} = h^2 f_{i,j}$$

Ogni punto interno dipende solo dai **4 vicini cardinali** (sopra, sotto, sinistra, destra) — questo è lo stencil 5-point.

### 2.4 Formula iterativa di Jacobi

Risolvendo rispetto a $u_{i,j}$:

$$u_{i,j}^{(k+1)} = \frac{1}{4}\left(u_{i+1,j}^{(k)} + u_{i-1,j}^{(k)} + u_{i,j+1}^{(k)} + u_{i,j-1}^{(k)} + h^2 f_{i,j}\right)$$

**Proprietà chiave:** ogni $u_{i,j}^{(k+1)}$ si calcola usando **solo i valori della vecchia iterazione** $k$. Questo garantisce che tutti i punti interni possano essere aggiornati **in parallelo**, rendendo il metodo intrinsecamente adatto alla GPU.

### 2.5 Criterio di convergenza

L'errore a ogni iterazione è la **norma infinito della differenza** tra due iterazioni consecutive:

$$\text{error}^{(k)} = \max_{i,j} \left|u_{i,j}^{(k+1)} - u_{i,j}^{(k)}\right|$$

Ci si ferma quando `error < tol` (default: `1e-7`).

### 2.6 Doppio buffer

Per implementare Jacobi servono **due griglie** `u` (vecchia) e `u_new` (nuova). Alla fine di ogni iterazione i puntatori vengono **scambiati** (`swap`), evitando costose copie di memoria.

---

## 3. Obiettivi Minimi della Traccia e Verifica del Rispetto

La traccia (pag. 8, slide 24/135) elenca **7 obiettivi minimi**. Segue la verifica punto per punto.

| # | Obiettivo dalla traccia | Dove implementato | ✅/⚠️ |
|---|---|---|---|
| 1 | Implementare Jacobi 2D su CPU | `src/cpu_solver.cpp` — `jacobi_step_cpu()` + `jacobi_cpu()` | ✅ |
| 2 | Implementare Jacobi 2D su GPU | `src/gpu_solver.cu` — tre varianti kernel | ✅ |
| 3 | Calcolare una misura di convergenza | max\|u_new − u\| ad ogni iterazione | ✅ |
| 4 | Implementare una riduzione su GPU per residuo/norma | `src/reduction.cu` (standalone) + riduzione embedded in V2/V3 | ✅ |
| 5 | Validare il risultato confrontando CPU e GPU | `src/validation.cpp` — `print_validation()` | ✅ |
| 6 | Confrontare tempo per iterazione e tempo totale | `SolverResult.ms_per_iter` + `SolverResult.total_ms` | ✅ |
| 7 | Confrontare prestazioni al variare di griglia e iterazioni | Benchmark suite in `main.cu` con N ∈ {128, 256, 512, 1024} | ✅ |

**Tutti e 7 gli obiettivi minimi sono rispettati e superati** (il progetto aggiunge ulteriori estensioni non richieste, come V3 coalesced e la validazione contro la soluzione analitica esatta).

### Strategie per il controllo della convergenza (slide 35/135)

La traccia descrive due strategie:

- **Strategia A** (semplice, meno efficiente): calcolare l'errore ogni tot su CPU → implementata in **V1** (`jacobi_gpu_naive`).
- **Strategia B** (efficiente, con riduzione GPU): ogni blocco calcola il suo max locale → implementata in **V2** e **V3**.

Entrambe le strategie sono implementate correttamente.

### Concetto di doppio buffer (slide 25/135)

La traccia richiede esplicitamente l'uso del doppio buffer con swap dei puntatori. Questo è implementato in tutte le versioni:

```cpp
// Swap device pointers (nessuna copia!)
double* tmp = d_u_new;
d_u_new = d_u;
d_u     = tmp;
```

---

## 4. Architettura del Progetto — Mappa del Codice

```
MiniSolver_2D_Poisson/
│
├── main.cu                          ← Punto di ingresso; orchestrazione di tutto
│
├── include/                         ← Dichiarazioni, configurazioni e interfacce globali
│   ├── common.h                     ← Macro (IDX, TILE_*, CUDA_CHECK) e strutture dati comuni (Params/Result)
│   ├── timer.h                      ← Classi di temporizzazione CPU (chrono) e GPU (cudaEvent) con compilazione condizionale
│   ├── cpu_solver.h                 ← Firme delle funzioni per il risolutore seriale CPU
│   ├── reduction.h                  ← Firme per la riduzione parallela standalone su GPU
│   ├── gpu_solver.h                 ← Firme dei kernel e dei wrapper per le tre varianti GPU e benchmark
│   └── validation.h                 ← Firme delle funzioni di confronto e misurazione dell'errore analitico
│
└── src/                             ← File sorgente contenenti le implementazioni
    ├── cpu_solver.cpp               ← Inizializzazione griglia e ciclo di risoluzione seriale CPU
    ├── reduction.cu                 ← Implementazione del kernel di riduzione massimo e del relativo wrapper host
    ├── gpu_solver.cu                ← Implementazione dei kernel GPU (V1/V2/V3), dei cicli di controllo e benchmark
    └── validation.cpp               ← Calcoli statistici dell'errore (RMS, max diff) e soluzione analitica
```

### Dettagli del Design Architetturale

1. **Separazione Netta CPU / GPU**: I file sorgente `.cpp` contengono puro codice C++ standard destinato alla compilazione host, mentre i sorgenti `.cu` contengono costrutti ed estensioni CUDA destinati alla compilazione tramite il compilatore NVIDIA `nvcc`.
2. **Accoppiamento Tramite Strutture Dati**: La comunicazione tra l'orchestratore (`main.cu`) e i moduli di risoluzione avviene esclusivamente mediante le strutture `SolverParams` (per i parametri di input come griglia $N$, tolleranza, iterazioni e frequenza di controllo) e `SolverResult` (per la restituzione di statistiche quali iterazioni totali, errore residuo finale e tempi di esecuzione). Questo impedisce l'uso di variabili globali e garantisce l'uniformità dei test.
3. **Compilazione Condizionale nei File Header**: File come __timer.h__ utilizzano la macro predefinita di sistema `__CUDACC__`. Ciò consente di nascondere le implementazioni basate su API CUDA (come `cudaEvent_t`) quando il file viene incluso in file C++ standard privi di supporto CUDA (compilati con compilatori classici come GCC, Clang o MSVC), evitando errori in fase di linking.


### Dipendenze tra moduli

```
main.cu
  ├── common.h           (macro, struct)
  ├── timer.h            (CpuTimer)
  ├── cpu_solver.h/cpp   (CPU solver)
  ├── validation.h/cpp   (validazione)
  ├── reduction.h/cu     (riduzione standalone)
  └── gpu_solver.h/cu    (GPU solver: usa common.h, timer.h)
        └── reduction.h/cu  (riduzione interna embedded per V2/V3)
```

### Flusso di dati principale

```
initialize(h_u, h_f)
      │
      ├──► jacobi_cpu(h_u)           → cpu_res
      │
      ├──► cudaMemcpy(h→d)
      │         │
      │    jacobi_gpu_naive(d_u)     → v1_res   [Strategia A: copia H↔D ogni check_every iter]
      │    jacobi_gpu_optimized(d_u) → v2_res   [Strategia B: d_block_max, 8KB vs N²*8B]
      │    jacobi_gpu_coalesced(d_u) → v3_res   [Strategia B + loader coalesced]
      │         │
      │    cudaMemcpy(d→h)
      │
      ├──► print_validation(cpu vs gpu_v1, v2, v3)
      ├──► max_error_vs_exact(h_u_gpu, N, h)
      │
      └──► Benchmark suite: N ∈ {128, 256, 512, 1024}, 1000 iter/versione
```

---

## 5. Analisi File per File

### 5.1 `include/common.h`

**Ruolo:** Header di configurazione globale — tutto il progetto include questo file.

**Contenuto:**

```cpp
#define IDX(i, j, N)  ((i) * (N) + (j))   // Linearizzazione row-major
#define TILE_X  16                        // Thread per blocco in direzione x (colonne)
#define TILE_Y  16                        // Thread per blocco in direzione y (righe)
#define SMEM_X  (TILE_X + 2)              // 18: tile + 1 cella halo a sinistra + 1 a destra
#define SMEM_Y  (TILE_Y + 2)              // 18: tile + 1 cella halo sopra + 1 sotto
#define DEFAULT_CHECK_EVERY  100          // Ogni quante iterazioni si controlla la convergenza
```

**`CUDA_CHECK(call)`:** Macro per il controllo degli errori CUDA. Ogni chiamata CUDA runtime viene avvolta in questa macro — in caso di errore stampa file, linea e messaggio, poi termina con `exit(EXIT_FAILURE)`.

**`SolverParams`:** Struttura che raggruppa tutti i parametri del solver:
- `N`: dimensione griglia ($N \times N$)
- `h`: spaziatura = $1/(N-1)$
- `tol`: tolleranza di convergenza
- `max_iter`: limite iterazioni
- `check_every`: frequenza di controllo convergenza

**`SolverResult`:** Struttura restituita da ogni solver:
- `iters`: iterazioni effettuate
- `final_error`: ultimo errore calcolato
- `total_ms`: tempo totale in millisecondi
- `ms_per_iter`: tempo medio per iterazione

**Perché `TILE_X = TILE_Y = 16`?**
- 16×16 = 256 thread per blocco. È un multiplo di 32 (dimensione del warp) → nessuno spreco di thread in un warp. 
- 256 thread per blocco è un numero ottimale per molte GPU NVIDIA: permette buona occupancy senza eccedere il limite di 1024 thread/blocco.
- Con N multiplo di 16 si evitano blocchi parziali al bordo.

---

### 5.2 `include/timer.h`

**Ruolo:** Misura del tempo — due classi, una per CPU e una per GPU.

**`CpuTimer`:** Usa `std::chrono::high_resolution_clock`. Adatta per misurare il tempo del solver CPU e del benchmark standalone.

**`CudaTimer`:** Usa `cudaEvent_t`. Questa è la scelta corretta per la GPU:
- `cudaEventRecord()` inserisce un evento nella coda dei comandi CUDA.
- `cudaEventSynchronize()` attende che la GPU abbia completato tutti i comandi fino all'evento `stop`.
- `cudaEventElapsedTime()` misura il tempo trascorso tra i due eventi **sulla GPU**, eliminando la latenza di sincronizzazione che affliggerebbe un timer CPU.

**Perché non usare `clock()` o `std::chrono` per la GPU?**  
Le operazioni CUDA sono **asincrone**: un timer CPU misurebbe il tempo di lancio del kernel, non il tempo di esecuzione reale sulla GPU. `cudaEventElapsedTime` è immune a questo problema perché gli eventi sono accodati direttamente nella GPU pipeline.

Il blocco `#ifdef __CUDACC__` garantisce che `CudaTimer` sia compilato solo da `nvcc` (non da compilatori C++ normali), evitando errori di compilazione quando il file è incluso da `.cpp` puro.

---

### 5.3 `include/cpu_solver.h`

**Ruolo:** Dichiarazioni pubbliche del modulo CPU solver.

Contiene documentazione Doxygen per tutte le funzioni:
- `initialize()`: inizializza `u` (tutto a zero) e `f` (termine sorgente $2\pi^2 \sin(\pi x)\sin(\pi y)$)
- `jacobi_step_cpu()`: esegue uno sweep di Jacobi e restituisce `max|u_new - u|`
- `jacobi_cpu()`: loop principale Jacobi con timer e criterio di arresto

---

### 5.4 `src/cpu_solver.cpp`

**Ruolo:** Implementazione del solver Jacobi seriale su CPU.

#### `initialize(double* u, double* f, int N, double h)`

```cpp
for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
        const double x = j * h;   // x → direzione delle colonne
        const double y = i * h;   // y → direzione delle righe
        u[IDX(i, j, N)] = 0.0;
        f[IDX(i, j, N)] = 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
    }
}
```

- Inizializza `u = 0` ovunque (condizioni di Dirichlet sul bordo già soddisfatte).
- Inizializza `f` con il termine sorgente analitico. Il fattore $2\pi^2$ deriva dalla scelta $u_{\text{exact}} = \sin(\pi x)\sin(\pi y)$: applicando il Laplaciano negativo si ottiene esattamente $2\pi^2 \sin(\pi x)\sin(\pi y)$.

#### `jacobi_step_cpu(double* u_new, const double* u, const double* f, int N, double h2)`

```cpp
for (int i = 1; i < N-1; ++i) {       // salta bordo superiore e inferiore
    for (int j = 1; j < N-1; ++j) {   // salta bordo sinistro e destro
        const int id = IDX(i, j, N);
        const double val = 0.25 * (
            u[IDX(i-1, j, N)] +   // top
            u[IDX(i+1, j, N)] +   // bottom
            u[IDX(i, j-1, N)] +   // left
            u[IDX(i, j+1, N)] +   // right
            h2 * f[id]
        );
        u_new[id] = val;
        const double diff = fabs(val - u[id]);
        if (diff > error) error = diff;   // traccia il massimo
    }
}
```

Corrisponde **esattamente** allo pseudocodice della traccia (slide 26/135). La formula del Jacobi step è:

$$u_{\text{new}}[i,j] = 0.25 \cdot (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + h^2 \cdot f[i,j])$$

#### `jacobi_cpu(SolverParams params, double* u, double* u_new, const double* f)`

- Loop `while (iter < maxi)`: corrisponde al ciclo principale della traccia (slide 32/135).
- Ogni 500 iterazioni stampa il progresso.
- Swap dei puntatori: `double* tmp = u_new; u_new = u; u = tmp;` → **zero copie di dati**.
- Restituisce `SolverResult` con statistiche complete.

---

### 5.5 `include/reduction.h`

**Ruolo:** Dichiarazioni della riduzione parallela standalone su GPU.

Questo header definisce l'interfaccia pubblica per l'algoritmo di riduzione per il calcolo del valore massimo di un array lineare. La separazione del codice di riduzione in un modulo autonomo permette di eseguire test di correttezza isolati (Unit Testing) ed evita di appesantire il modulo del solutore GPU con funzioni helper generiche.

#### Struttura delle funzioni dichiarate:
1. **`reduce_max_kernel`**:
   - È la dichiarazione del kernel CUDA (`__global__`) che esegue la riduzione parallela dei dati.
   - Accetta in input `input` come puntatore a sola lettura `const double* __restrict__` per favorire le ottimizzazioni di compilazione (come il caching nei registri e l'uso di cache L1/L2 read-only).
   - Accetta in output un array di output parziali `double* __restrict__ output` la cui dimensione corrisponde al numero totale di blocchi lanciati nella griglia. Ciascun blocco calcola e scrive il proprio massimo locale all'indice `blockIdx.x`.
   - Richiede l'allocazione dinamica di memoria condivisa (`extern __shared__ double sdata[]`) specificata in byte al lancio del kernel.
2. **`reduce_max_gpu`**:
   - È la funzione wrapper host (`extern "C"` o standard C++) chiamata dalla CPU.
   - Incapsula tutta la gestione della memoria sulla GPU (allocazione temporanea con `cudaMalloc`, copie di dati con `cudaMemcpy`, e deallocazione con `cudaFree`).
   - Gestisce la configurazione dei blocchi e il lancio di uno o più passi del kernel a seconda della dimensione dei dati.
   - Garantisce la sincronizzazione dei calcoli tramite `cudaDeviceSynchronize()` prima di restituire il risultato finale all'host come singolo valore scalare di tipo `double`.

---

### 5.6 `src/reduction.cu`

**Ruolo:** Implementazione della riduzione parallela massimo su GPU — modulo standalone.

#### 5.6.1 L'operazione di Riduzione Parallela (Teoria e Sfide)
Nel calcolo parallelo, una **riduzione** (Reduction) è un'operazione che trasforma un insieme (o array) di $N$ elementi in un singolo valore scalare applicando un operatore binario associativo (e opzionalmente commutativo). Esempi comuni includono la somma, il prodotto, il minimo o, come in questo caso, il massimo matematico.

- **Complessità Seriale**: Su CPU, la riduzione si implementa con un banale ciclo sequenziale in tempo $O(N)$ ed uso di memoria $O(1)$.
- **Complessità Parallela**: Su GPU, un ciclo sequenziale comporterebbe conflitti di scrittura (race conditions) o la serializzazione dei thread se si usassero operazioni atomiche globali (`atomicMax`). Per sfruttare la massiva architettura parallela delle GPU, si implementa un algoritmo basato su un **albero binario** (Binary Tree Reduction) in cui il numero di elementi attivi viene dimezzato ad ogni passo. La complessità temporale si riduce così a $O(\log N)$.
- **Sfide CUDA**: Una riduzione efficiente deve minimizzare l'overhead di sincronizzazione dei blocchi, evitare i conflitti nei banchi di memoria condivisa (shared memory bank conflicts) e massimizzare il throughput di memoria globale evitando accessi non coalescenti.

```
Fase di Riduzione ad Albero Binario in Shared Memory (per un blocco):
Passo 0 (s=8): [ x0  x1  x2  x3  x4  x5  x6  x7  x8  x9  x10 x11 x12 x13 x14 x15 ]
               │   │   │   │   │   │   │   │  ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲
               └───┼───┼───┼───┼───┼───┼───┼──┘   │   │   │   │   │   │   │
                   └───┼───┼───┼───┼───┼───┼──────┘   │   │   │   │   │   │
                       └─...   │   │   │   │          │   │   │   │   │   │
Passo 1 (s=4): [  m0  m1  m2  m3  m4  m5  m6  m7 ]   (m_i = max(x_i, x_{i+8}))
               │   │   │   │  ▲   ▲   ▲   ▲
               └───┼───┼───┼──┘   │   │   │
                   └─...          └─..│...│...
Passo 2 (s=2): [   k0  k1  k2  k3 ]
Passo 3 (s=1): [   w0  w1 ]  --> [ Max Globale del Blocco ]
```

#### 5.6.2 Analisi dei Componenti del Codice

##### A) `warpReduceMax(double val)` — device helper
Questa funzione esegue la riduzione del massimo all'interno di un singolo **warp** (gruppo di 32 thread che eseguono istruzioni in modalità lockstep).

```cuda
__device__ __forceinline__
double warpReduceMax(double val) {
    unsigned int mask = 0xffffffff;    // Tutti i 32 thread del warp partecipano
    val = fmax(val, __shfl_down_sync(mask, val, 16));
    val = fmax(val, __shfl_down_sync(mask, val,  8));
    val = fmax(val, __shfl_down_sync(mask, val,  4));
    val = fmax(val, __shfl_down_sync(mask, val,  2));
    val = fmax(val, __shfl_down_sync(mask, val,  1));
    return val;
}
```

**Come funziona `__shfl_down_sync`:**  
`__shfl_down_sync(mask, val, offset)` permette a un thread di leggere il registro `val` di un altro thread a distanza `offset` **senza passare dalla shared memory**. Questo elimina le scritture/letture su SMEM e i relativi `__syncthreads()` per le ultime 5 iterazioni della riduzione binaria.

- **Round 1 (offset=16):** thread 0 prende max(lane0, lane16), thread 1 prende max(lane1, lane17), etc.
- **Round 2 (offset=8):** thread 0 prende max di lane0 e lane8, etc.
- **…fino a offset=1:** thread 0 ha il massimo di tutti i 32 thread del warp.

#### `reduce_max_kernel` — il kernel

   ```cuda
   int i = (int)(blockIdx.x * (blockDim.x * 2) + tid);
   double myMax = -DBL_MAX;
   if (i < n) myMax = input[i];
   if (i + blockDim.x < n) myMax = fmax(myMax, input[i + blockDim.x]);
   sdata[tid] = myMax;
   __syncthreads();
   ```
2. **Riduzione in Shared Memory (Strides > 32)**:
   Gli elementi vengono scritti nell'array condiviso `sdata[tid]`. Successivamente, un ciclo dimezza lo stride ad ogni iterazione:
   ```cuda
   for (int s = blockDim.x / 2; s > 32; s >>= 1) {
       if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
       __syncthreads();
   }
   ```
   La barriera `__syncthreads()` è fondamentale ad ogni iterazione per evitare corse critiche (race conditions) in cui un thread legge un dato prima che il suo vicino lo abbia aggiornato.
3. **Ottimizzazione per gli Ultimi 64 Elementi**:
   Quando lo stride scende a 32, i risultati parziali risiedono nei primi 64 elementi di `sdata`. A questo punto, solo il primo warp (thread 0-31) rimane attivo. Esso unisce gli elementi della metà superiore in quella inferiore ed esegue `warpReduceMax(v)`. Il thread 0 del blocco scrive il massimo finale del blocco in memoria globale:
   ```cuda
   if (tid < 32) {
       double v = fmax(sdata[tid], sdata[tid + 32]);
       v = warpReduceMax(v);
       if (tid == 0) output[blockIdx.x] = v;
   }
   ```

##### C) `reduce_max_gpu` — il wrapper host (Algoritmo a Due Fasi)
La riduzione parallela su GPU non può sincronizzare l'intera griglia globalmente all'interno di un singolo kernel (mancanza di barriere hardware globali tra blocchi diversi). Per questo motivo, la riduzione di array di grandi dimensioni viene spezzata in due fasi distinte gestite dall'host:

- **Fase 1 (Block Reduction)**: Viene lanciato il kernel configurando $B$ blocchi. Ognuno elabora una porzione dell'array globale di input e scrive un massimo parziale. L'output è un array temporaneo `d_partial` di dimensione $B$.
  ```cuda
  const int threads = 256;
  const int blocks  = (n + threads * 2 - 1) / (threads * 2);
  cudaMalloc(&d_partial, blocks * sizeof(double));
  reduce_max_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_data, d_partial, n);
  ```
- **Fase 2 (Global Reduction)**: 
  - Se $B = 1$ (l'array originale era piccolo), il massimo globale coincide con `d_partial[0]` che viene copiato direttamente sull'host.
  - Se $B > 1$, viene eseguito un secondo lancio del kernel con **un solo blocco** (`gridDim.x = 1`) e $B$ elementi in ingresso. Questo blocco unico riduce i massimi parziali calcolati nella prima fase producendo il valore massimo globale definitivo, che viene poi copiato sull'host.
  ```cuda
  if (blocks > 1) {
      reduce_max_kernel<<<1, threads, threads * sizeof(double)>>>(d_partial, d_final, blocks);
  }
  ```
  La shared memory dinamica (`extern __shared__ double sdata[]`) è passata come terzo argomento al lancio del kernel: `threads * sizeof(double)`.


---

### 5.7 `include/gpu_solver.h`

**Ruolo:** Dichiarazioni delle tre varianti kernel + wrapper host + benchmark.

Documenta in modo esaustivo:
- L'analisi di coalescenza degli accessi in memoria
- Il layout della shared memory (`s_u` con halo, `s_max` per la riduzione)
- Le tre varianti:
  - **V1 `jacobi_kernel_naive`:** pura global memory, Strategia A
  - **V2 `jacobi_kernel_shared`:** shmem con halo, Strategia B
  - **V3 `jacobi_kernel_shared_coalesced`:** loader coalesced ottimizzato, Strategia B

---

### 5.8 `src/gpu_solver.cu`

**Il file più importante del progetto.** Contiene tutte le implementazioni GPU.

#### V1 — `jacobi_kernel_naive`

```cuda
const int j = blockIdx.x * blockDim.x + threadIdx.x;   // colonna (fast index)
const int i = blockIdx.y * blockDim.y + threadIdx.y;   // riga
if (i < 1 || i > N-2 || j < 1 || j > N-2) return;    // guard boundary

u_new[IDX(i,j,N)] = 0.25 * (
    u[IDX(i-1,j,N)] + u[IDX(i+1,j,N)] +
    u[IDX(i,j-1,N)] + u[IDX(i,j+1,N)] +
    h2 * f[IDX(i,j,N)]
);
```

**Mapping thread→griglia:**
- `threadIdx.x` → colonna `j` (dimensione veloce, contigua in memoria)
- `threadIdx.y` → riga `i`

Questo garantisce che i thread di uno stesso warp accedano a indirizzi contigui per i carichi orizzontali (sinistra/destra), massimizzando la **coalescenza**.

**Coalescenza in V1:**
- Lettura di `u[i,j]`: stride-0 → perfettamente coalesced
- Lettura di `u[i,j±1]`: stride ±1 → quasi-coalesced (shift di 1 double)
- Lettura di `u[i±1,j]`: stride N → **non coalesced** (accesso verticale), ma diversi thread in un warp accedono a righe diverse → non causa serializzazione

**Convergenza (Strategia A):** ogni `check_every` iterazioni, copia entrambe le griglie su host e calcola `max|diff|` su CPU. Costoso per N grande (2 × N² × 8 byte per check), ma semplice.

#### V1 — `jacobi_kernel_naive`

```cuda
const int j = blockIdx.x * blockDim.x + threadIdx.x;   // colonna (fast index)
const int i = blockIdx.y * blockDim.y + threadIdx.y;   // riga
if (i < 1 || i > N-2 || j < 1 || j > N-2) return;    // guard boundary

u_new[IDX(i,j,N)] = 0.25 * (
    u[IDX(i-1,j,N)] + u[IDX(i+1,j,N)] +
    u[IDX(i,j-1,N)] + u[IDX(i,j+1,N)] +
    h2 * f[IDX(i,j,N)]
);
```

- **Mapping thread $\to$ griglia**: La disposizione bidimensionale dei thread mappa `threadIdx.x` sulla coordinata delle colonne `j` (che varia più velocemente in memoria secondo il layout row-major del C++) e `threadIdx.y` sulla coordinata delle righe `i`. Questo assicura che thread consecutivi nello stesso warp (che differiscono per `threadIdx.x`) accedano ad elementi consecutivi in memoria globale (`u[IDX(i, j)]`, `u[IDX(i, j+1)]`, etc.), consentendo la **coalescenza della memoria** (Memory Coalescing).
- **Analisi degli accessi alla memoria globale**:
  - **Scrittura di `u_new[IDX(i,j,N)]`**: Sequenziale e allineata $\to$ coalescenza ideale (1 transazione da 128 byte per ogni mezza riga di warp).
  - **Lettura di `u[IDX(i,j±1,N)]`**: Spostata di $\pm 8$ byte rispetto al centro. Nonostante il disallineamento, l'hardware unifica la richiesta in un numero minimo di transazioni di memoria globale (1 o 2 transazioni).
  - **Lettura di `u[IDX(i±1,j,N)]`**: Accesso non coalescente perché la distanza in memoria tra i thread è pari a $N$ elementi. I 32 thread di un warp inviano 32 richieste separate a indirizzi distanti. Tuttavia, grazie all'**L1/L2 Cache** e alla cache a sola lettura delle moderne GPU, gli elementi vicini caricati per le righe adiacenti vengono conservati in cache e riutilizzati dai blocchi verticalmente adiacenti, mitigando l'impatto prestazionale.
- **Dettagli sulla convergenza (Strategia A)**:
  La Strategia A esegue il controllo scaricando l'intera griglia $N \times N$ tramite `cudaMemcpy` dalla GPU all'host ogni `check_every` iterazioni. La CPU calcola poi la norma infinito con un ciclo seriale. Per griglie grandi (es. $1024 \times 1024$), questa operazione costituisce un grave **collo di bottiglia**:
  - Trasferimento di $2 \times N^2 \times 8 \text{ byte} = 16 \text{ MB}$ su bus PCIe (tipicamente limitato a $\approx 12\text{-}16 \text{ GB/s}$).
  - Il tempo di CPU per calcolare la differenza ($O(N^2)$ operazioni seriali in virgola mobile) interrompe la pipeline della GPU, degradando pesantemente le prestazioni complessive.

#### V2 — `jacobi_kernel_shared`

Questa variante ottimizza gli accessi alla memoria globale memorizzando la griglia locale (tile) in **Shared Memory** (memoria su chip a bassissima latenza, paragonabile ai registri).

- **Schema della shared memory**:
  ```
      s_u [SMEM_Y][SMEM_X]  =  s_u[18][18]  = 324 double
      
      Layout (indici shared memory):
      ┌─────────────────────────────────┐
      │ halo top:     s_u[0][1..16]     │  ← 16 thread (ty==0) caricano
      │ ┌───────────────────────────┐   │
      │ │ tile:   s_u[1..16][1..16] │   │  ← tutti i 256 thread caricano
      │ └───────────────────────────┘   │
      │ halo bot:     s_u[17][1..16]    │  ← 16 thread (ty==15) caricano
      │ halo sx:      s_u[1..16][0]     │  ← 16 thread (tx==0)  caricano
      │ halo dx:      s_u[1..16][17]    │  ← 16 thread (tx==15) caricano
      └─────────────────────────────────┘
  ```
  La tile ha dimensione $18 \times 18$ per contenere gli elementi interni ($16 \times 16$) più una corona circolare (halo) di spessore pari a 1 cella, necessaria per calcolare lo stencil a 5 punti senza dover accedere alla memoria globale.
- **Warp Divergence e accessi non coalescenti nel caricamento di V2**:
  Il caricamento dell'halo in V2 avviene tramite istruzioni condizionali `if (ty == 0)`, `if (tx == 0)`, etc. Questo approccio presenta due inefficienze:
  - **Divergenza del Warp (Warp Divergence)**: All'interno di uno stesso warp, solo una parte dei thread soddisfa la condizione `if` (es. solo i thread con `tx == 0` sul bordo sinistro). Gli altri thread del warp rimangono inattivi durante l'esecuzione di quel ramo, sprecando cicli hardware.
  - **Accessi Strided**: I thread di bordo sinistro/destro (`tx == 0` e `tx == 15`) caricano dati verticalmente con stride $N$ in memoria globale, generando carichi non coalescenti.
- **Calcolo del Risparmio Teorico di Bandwidth**:
  Per un blocco di $16 \times 16 = 256$ elementi computazionali:
  - In **V1 (Naive)**: Ogni thread legge 4 vicini e il valore corrente dal buffer globale, per un totale di $256 \times 5 = 1280$ letture globali.
  - In **V2 (Shared)**: I 256 thread caricano una sola volta i propri elementi al centro, e solo i thread di bordo caricano l'halo ($4 \times 16 = 64$ carichi aggiuntivi). Il numero totale di letture globali scende a $256 + 64 + 256 \text{ (per f)} = 576$ letture.
  - **Riduzione del traffico di memoria**: $\approx 55\%$ di risparmio sulla banda passante di lettura.
- **Riduzione Embedded in Shared Memory (Strategia B)**:
  La convergenza viene calcolata riducendo le differenze assolute locali all'interno di ogni blocco tramite shared memory:
  ```cuda
  s_max[tid] = diff;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
      if (tid < stride) {
          s_max[tid] = fmax(s_max[tid], s_max[tid + stride]);
      }
      __syncthreads();
  }
  ```
  Alla fine del ciclo, il thread 0 scrive il massimo parziale del blocco nell'array globale `d_block_max`. Invece di copiare $N^2$ elementi, l'host copia solo un vettore di dimensione pari al numero di blocchi (es. per $N=512$, la griglia ha $32 \times 32 = 1024$ blocchi. Il trasferimento host-device si riduce da $2 \text{ MB}$ a soli $8 \text{ KB}$, un abbattimento di circa **250 volte** del traffico su PCIe).

#### V3 — `jacobi_kernel_shared_coalesced`

Risolve le inefficienze di caricamento presenti in V2 eliminando la divergenza dei warp e rendendo i carichi dell'halo completamente coalescenti.

- **Tecnica di Caricamento Lineare Coalescente**:
  Invece di far caricare l'halo ai singoli thread di bordo tramite rami condizionali divergenti, i 256 thread del blocco collaborano per caricare l'intero array bidimensionale di shared memory come se fosse un vettore monodimensionale contiguo di $18 \times 18 = 324$ elementi:
  - **Fase 1**: I thread 0..255 caricano linearmente i primi 256 elementi.
  - **Fase 2**: I thread 0..67 caricano i restanti 68 elementi ($324 - 256 = 68$).
- **Mappatura da indice 1D shared a coordinate 2D globali**:
  Il codice calcola matematicamente le coordinate di shared memory (`li`, `lj`) e le proietta sulle coordinate globali (`gi`, `gj`), gestendo i bordi fisici del dominio:
  ```cuda
  const int linear_id = tid;
  const int li = linear_id / SMEM_X;
  const int lj = linear_id % SMEM_X;
  const int gi = (int)blockIdx.y * TILE_Y + li - 1; // offset -1 per halo sinistro/superiore
  const int gj = (int)blockIdx.x * TILE_X + lj - 1;
  ```
- **Vantaggio Hardware**:
  Poiché i thread consecutivi di un warp accedono a indici `linear_id` consecutivi, le letture in memoria globale puntano a coordinate `gj` consecutive. Di conseguenza, i carichi per l'intero blocco (incluso l'halo) risultano perfettamente coalescenti. La GPU unifica i trasferimenti in transazioni di cache line da 128 byte, massimizzando il throughput della memoria globale.


---

### 5.9 `include/validation.h`

**Ruolo:** Dichiarazioni delle funzioni di validazione numerica.

Tre metriche di confronto:
- `max_abs_diff(a, b, n)`: norma infinito `max|a[k] - b[k]|`
- `rms_diff(a, b, n)`: norma L2 (RMSE) `sqrt(sum(a[k]-b[k])^2 / n)`
- `max_error_vs_exact(u, N, h)`: confronto con soluzione analitica esatta

---

### 5.10 `src/validation.cpp`

**Ruolo:** Implementazione delle funzioni di validazione.

#### `exact_at(int i, int j, double h)`

```cpp
return std::sin(M_PI * j * h) * std::sin(M_PI * i * h);
```

Calcola la soluzione esatta $u_{\text{exact}}(x,y) = \sin(\pi x)\sin(\pi y)$ nel punto $(j \cdot h, i \cdot h)$.

#### `max_error_vs_exact`

Itera su tutti gli N×N punti e calcola `max|u[i,j] - u_exact(i,j)|`. Include i punti di bordo che devono valere 0.

#### `print_validation`

Confronta due soluzioni numeriche (es. CPU vs GPU V1). Il criterio di pass/fail usa `tol * 100.0` invece di `tol` — questo è intentenzionale: le due soluzioni (CPU e GPU) differiscono per l'ordine in cui vengono eseguite le operazioni floating point, e la differenza si accumula nel corso delle iterazioni, quindi una tolleranza 100 volte più larga è ragionevole.

---

### 5.11 `main.cu`

**Ruolo:** Orchestrazione completa del programma.

Questo modulo rappresenta il punto di ingresso dell'applicazione (Entry Point) e coordina l'intera pipeline di esecuzione: parsing dei parametri di input, configurazione hardware della GPU, allocazione e inizializzazione delle risorse host/device, esecuzione dei risolutori, validazione scientifica dei risultati e benchmarking finale delle varianti.

#### `print_gpu_properties()`
Stampa in modo strutturato le caratteristiche fisiche della GPU rilevata a runtime tramite le API CUDA (`cudaGetDeviceProperties`). Questo è fondamentale per documentare l'ambiente di esecuzione e comprendere i vincoli hardware (es. memoria condivisa disponibile, warp size e limiti sui thread per blocco).

#### `test_standalone_reduction()`
Costituisce un test di unità (Unit Test) per verificare l'accuratezza del riduttore parallelo prima del suo inserimento nei risolutori completi. Genera un vettore sintetico di 2.5 milioni di elementi, forza un valore massimo noto (9999.87654) ad un indice casuale, esegue il calcolo seriale su CPU ed esegue `reduce_max_gpu()`. Verifica che la differenza assoluta sia inferiore alla tolleranza di macchina ($10^{-9}$), stampando l'esito `[ PASS ]` o `[ FAIL ]`.

#### `main()` — Gestione dei Parametri da Linea di Comando
Il programma accetta diversi argomenti per configurare l'esecuzione:
- `--n <int>`: Dimensione della griglia bidimensionale $N \times N$ (default: 256).
- `--tol <double>`: Tolleranza per il criterio di convergenza (default: $1.0\times 10^{-7}$).
- `--max-iter <int>`: Numero massimo di iterazioni concesse (default: 50000).
- `--check-every <int>`: Intervallo di iterazioni tra i controlli di convergenza (default: 100).
- `--no-scalability`: Disabilita la suite di benchmark finale.
- `--no-cpu`: Salta la baseline CPU seriale (utile per griglie grandi come $N > 512$ dove la CPU impiega tempi proibitivi).

#### Guardia di Allineamento della Griglia
```cpp
if (N % TILE_X != 0 || N % TILE_Y != 0) {
    N = ((N + TILE_X - 1) / TILE_X) * TILE_X;
}
```
Questa operazione di arrotondamento per eccesso al successivo multiplo intero della dimensione del blocco ($16$) è di fondamentale importanza:
1. Garantisce che i blocchi di thread coprano esattamente l'intero dominio computazionale senza generare parzializzazioni o thread inattivi sui bordi.
2. Semplifica la logica di indirizzamento della memoria condivisa in V2 e V3, prevenendo accessi fuori intervallo (Out-of-Bounds memory accesses) durante il caricamento cooperativo dell'halo.

#### Gestione dei Puntatori dopo lo Swap (Double-Buffering Copy-Back)
Poiché l'algoritmo di Jacobi scambia i puntatori di lettura e scrittura ad ogni iterazione per evitare la copia fisica dei dati, la soluzione finale al termine del ciclo `while` può risiedere nel buffer originale `d_u` o in quello ausiliario `d_u_new`, a seconda della parità delle iterazioni effettivamente compiute:
- Se il numero di iterazioni `iters` è **dispari**, il risultato finale corretto risiede nel buffer `d_u_new`.
- Se `iters` è **pari**, il risultato si trova in `d_u`.
Il codice in `main.cu` implementa questa selezione tramite l'operatore ternario prima di invocare la `cudaMemcpy` per il trasferimento dei risultati verso l'host:
```cpp
CUDA_CHECK(cudaMemcpy(h_u_gpu_v1, (v1_res.iters % 2 == 1) ? d_u_new : d_u, bytes, cudaMemcpyDeviceToHost));
```

#### La Suite di Benchmark di Scalabilità
Se abilitata, valuta il throughput dei risolutori su griglie crescenti $N \in \{128, 256, 512, 1024\}$ eseguendo esattamente 1000 iterazioni fisse senza controlli di convergenza. Questo isola le performance computazionali pure dai costi di trasferimento della riduzione.
Per ogni configurazione vengono calcolate tre metriche chiave:
1. **Time (ms)**: Tempo medio totale per 1000 iterazioni.
2. **MUpdates/s (Milioni di aggiornamenti al secondo)**: 
   $$\text{MUpdates/s} = \frac{(N-2) \times (N-2) \times 1000}{\text{Tempo (ms)} \times 1000}$$
   Rappresenta la velocità di aggiornamento delle celle interne del dominio.
3. **GB/s (Ideale)**: Misura la larghezza di banda di memoria effettivamente sfruttata, calcolata assumendo l'accesso ideale a 3 array (lettura di $u$, scrittura di $u_{\text{new}}$ e lettura di $f$):
   $$\text{GB/s} = \frac{3 \times \text{sizeof(double)} \times (N-2)^2 \times 1000}{\text{Tempo (ms)} \times 10^6}$$
4. **Speedup**: Rapporto tra il tempo della baseline CPU (opportunamente scalato per dimensioni elevate) e il tempo della variante GPU sotto test.


---

## 6. Linee Guida CUDA — Verifica Rispetto Best Practices

Il progetto è stato sviluppato seguendo rigorosamente le linee guida ufficiali di NVIDIA per l'ottimizzazione del codice CUDA (NVIDIA CUDA C++ Best Practices Guide), puntando a massimizzare l'efficienza computazionale e lo sfruttamento della banda di memoria.

### 6.1 Coalescenza degli Accessi in Memoria Globale ✅

- **Regola Hardware**: I controller di memoria delle GPU gestiscono i trasferimenti da e verso la memoria globale (VRAM) tramite transazioni allineate da **128 byte** (equivalenti a una cache line L2). Se i thread di un warp (32 thread) accedono a indirizzi di memoria contigui e allineati, l'intera richiesta viene servita da una singola transazione (Coalescing), massimizzando la banda passante.
- **Come rispettata nel progetto**:
  - Il mapping `threadIdx.x → colonna j` garantisce che thread adiacenti accedano a elementi `double` (8 byte) consecutivi in memoria. Un mezzo warp (16 thread) legge/scrive esattamente $16 \times 8 = 128$ byte contigui, saturando perfettamente una transazione di memoria.
  - Nella variante **V3**, il caricamento cooperativo lineare converte la memoria bidimensionale in un indice monodimensionale lineare `linear_id = tid`. I thread eseguono letture sequenziali e allineate da global memory anche per i dati dell'halo (comprese le righe superiore/inferiore e le colonne laterali), rimuovendo completamente gli accessi con stride $N$ che degradavano le performance in V2.

### 6.2 Uso della Shared Memory ✅

- **Regola Hardware**: Gli accessi alla memoria globale hanno latenze elevate ($\approx 400\text{-}800$ cicli di clock). La shared memory è una cache on-chip configurabile dall'utente con latenze estremamente ridotte ($\approx 20\text{-}30$ cicli). I dati riutilizzati all'interno dello stesso blocco di thread devono essere caricati in shared memory.
- **Come rispettata nel progetto**:
  - Nelle versioni **V2** e **V3**, l'intera porzione di griglia assegnata al blocco viene caricata in `s_u`. Nello sweep di Jacobi, ogni cella interna viene letta 4 volte per calcolare la media dei vicini cardinali dello stencil. Grazie alla shared memory, 3 di queste 4 letture avvengono su chip senza generare traffico verso la memoria globale.
  - L'halo aggiuntivo di 1 cella evita che i thread sul bordo del blocco debbano ricorrere a letture globali separate durante la fase di calcolo.

### 6.3 Sincronizzazione ed Evitamento di Race Conditions (`__syncthreads()`) ✅

- **Regola Hardware**: L'istruzione `__syncthreads()` definisce una barriera di esecuzione per tutti i thread appartenenti a un blocco. È indispensabile per prevenire pericoli di tipo Read-After-Write (RAW) o Write-After-Read (WAR) su dati condivisi.
- **Come rispettata nel progetto**:
  - Viene inserito un `__syncthreads()` subito dopo il caricamento cooperativo della tile in shared memory, prima che qualsiasi thread inizi il calcolo del Jacobi update.
  - Nella riduzione interna per il calcolo dell'errore (Strategia B), viene posto un `__syncthreads()` dopo la scrittura delle differenze locali in `s_max` e dopo ogni passo del ciclo di dimezzamento dello stride, impedendo corse critiche sui dati parziali.
  - Per evitare stalli hardware (deadlock), nessun `__syncthreads()` è inserito all'interno di rami condizionali divergenti (dove solo una frazione dei thread del blocco esegue l'istruzione).

### 6.4 Gestione e Controllo degli Errori su Chiamate CUDA ✅

- **Regola Hardware**: Molti errori in CUDA avvengono in modo asincrono (es. violazioni di memoria nei kernel rilevate solo al momento del successivo trasferimento o sincronizzazione). Non verificare gli errori può portare a comportamenti indefiniti difficili da diagnosticare.
- **Come rispettata nel progetto**:
  - La macro `CUDA_CHECK(call)` intercetta lo stato restituito da ogni chiamata runtime (es. `cudaMalloc`, `cudaMemcpy`, `cudaFree`). In caso di fallimento, estrae il codice errore con `cudaGetErrorString`, stampa file e riga, e interrompe l'esecuzione con `exit(EXIT_FAILURE)`.
  - Dopo ogni lancio di kernel viene eseguita la chiamata `CUDA_CHECK(cudaGetLastError())` per intercettare errori di configurazione o esecuzione immediata dei kernel.

### 6.5 Ottimizzazione dell'Occupancy e Dimensionamento dei Blocchi ✅

- **Regola Hardware**: L'Occupancy è il rapporto tra i warp attivi su un multiprocessore (SM) e il numero massimo di warp fisicamente supportabili. Un'occupancy elevata consente alla GPU di nascondere le latenze di memoria alternando l'esecuzione dei warp pronti.
- **Come rispettata nel progetto**:
  - La dimensione del blocco è impostata a $16 \times 16 = 256$ thread. Questo valore è un multiplo esatto della dimensione fisica del warp (32 thread), prevenendo lo spreco di risorse (warp parziali).
  - Un blocco di 256 thread garantisce un eccellente bilanciamento: non eccede il limite hardware di 1024 thread per blocco e permette allo scheduler di allocare molteplici blocchi sullo stesso SM, massimizzando l'occupancy teorica.

### 6.6 Riduzione Parallela Ottimizzata via Warp Shuffle (`__shfl_down_sync`) ✅

- **Regola Hardware**: Quando la riduzione binaria scende sotto i 32 elementi attivi, tutti i thread risiedono nello stesso warp. Usare shared memory e barriere `__syncthreads()` a questo livello introduce overhead inutili.
- **Come rispettata nel progetto**:
  - La funzione `warpReduceMax()` riduce l'array intra-warp tramite l'istruzione hardware `__shfl_down_sync`. Questa istruzione trasferisce i dati direttamente tra i registri fisici dei thread del warp senza passare per la shared memory e senza richiedere barriere di sincronizzazione, dimezzando i cicli di clock necessari nella fase conclusiva della riduzione.

### 6.7 Prevenzione dei Cold-Start Bias (Kernel Warmup) ✅

- **Regola Hardware**: Il primo lancio di un kernel CUDA comporta tempi aggiuntivi dovuti all'inizializzazione del runtime CUDA, alla compilazione JIT (Just-In-Time) dei driver e al popolamento iniziale delle cache. Effettuare benchmark su questo primo lancio altera significativamente le misurazioni.
- **Come rispettata nel progetto**:
  - La funzione `jacobi_gpu_benchmark` esegue una singola iterazione preliminare (Warmup) del kernel specifico, seguita da una sincronizzazione esplicita del dispositivo (`cudaDeviceSynchronize()`), prima di far partire il timer di misurazione reale.

### 6.8 Utilizzo dei Qualificatori `__restrict__` e Constant Caching ✅

- **Regola Hardware**: Il compilatore non può ottimizzare le letture da puntatori se esiste il rischio che i dati scritti da un puntatore sovrascrivano i dati letti da un altro (Pointer Aliasing).
- **Come rispettata nel progetto**:
  - Tutti i parametri dei kernel (`u_new`, `u`, `f`) sono marcati con `__restrict__`. Questo garantisce al compilatore che le aree di memoria non si sovrappongono.
  - Per i puntatori a sola lettura (come `u` e `f`), questo permette al compilatore `nvcc` di indirizzare le letture attraverso la cache globale a sola lettura (Read-Only Cache o cache LDG), ottimizzando i tempi di accesso.

### 6.9 Temporizzazione Accurata tramite Eventi CUDA ✅

- **Regola Hardware**: I lanci dei kernel su GPU sono asincroni rispetto all'host CPU. Utilizzare timer host (come `std::chrono` o `clock()`) senza sincronizzazione misura solo il tempo di accodamento del comando, non l'esecuzione fisica sulla GPU. Inserire sincronizzazioni continue degrada invece le prestazioni.
- **Come rispettata nel progetto**:
  - La classe `CudaTimer` gestisce la temporizzazione tramite gli eventi nativi CUDA (`cudaEvent_t`). Gli eventi vengono registrati direttamente nella coda dei comandi della GPU pipeline (`cudaEventRecord`) e il tempo trascorso viene calcolato dall'hardware con precisione al microsecondo tramite `cudaEventElapsedTime`, rimuovendo qualsiasi overhead o interferenza del sistema operativo host.

### 6.10 Assenza di Conflitti nei Banchi della Shared Memory (Bank Conflicts) ✅

- **Regola Hardware**: La shared memory è suddivisa in 32 moduli di memoria indipendenti ad accesso parallelo denominati **banchi** (Banks), con un'ampiezza di banda di 32 o 64 bit. Se thread diversi dello stesso warp accedono simultaneamente a indirizzi diversi che mappano sullo stesso banco di memoria, si verifica un **conflitto di banco** (Bank Conflict) e la GPU deve serializzare le richieste.
- **Come rispettata nel progetto**:
  - La larghezza della riga nella shared memory `s_u` è impostata tramite la macro `SMEM_X = TILE_X + 2 = 18`. Ciascun elemento è di tipo `double` (8 byte, 64 bit).
  - Nelle GPU moderne (architetture Kepler e successive), la memoria condivisa è organizzata in 32 banchi da 64 bit ciascuno. Poiché la riga ha dimensione 18 double, l'elemento `s_u[si][sj]` mappa sul banco `(si * 18 + sj) % 32`.
  - Durante l'accesso orizzontale sequenziale (`s_u[si][sj]` con `sj = threadIdx.x + 1`), thread adiacenti del warp accedono a colonne consecutive (`sj` e `sj+1`). Poiché la distanza tra elementi successivi è di 1 double, essi mappano su banchi consecutivi e differenti. Non si verifica alcun bank conflict.
  - Durante l'accesso verticale dello stencil (`s_u[si+1][sj]`), la riga successiva dista 18 elementi. La differenza di banco tra due thread adiacenti $T_0$ e $T_1$ che accedono rispettivamente a `s_u[si+1][0]` e `s_u[si+1][1]` è:
    $$\Delta \text{banco} = ((si+1) \times 18 + 1 - (si+1) \times 18) \pmod{32} = 1$$
    Gli accessi rimangono distribuiti su banchi diversi. Il design è **completamente esente da conflitti significativi**.


---

## 7. Analisi delle Tre Varianti GPU

### 7.1 Comparativa Tecnica

| Caratteristica | V1 (Naive) | V2 (Shared) | V3 (Coalesced) |
|---|---|---|---|
| Memoria compute | Global | Shared | Shared |
| Caricamento halo | — | Parziale (thread bordo) | Lineare (tutti thread) |
| Coalescenza load | Buona per orizzontale, mediocre per verticale | Buona | Ottima |
| Riduzione convergenza | Strategia A (copia N² double) | Strategia B (copia num_blocks double) | Strategia B |
| Global loads per punto | 5 (4 vicini + f) | ~1.5 (tile riusata) + f | ~1.3 + f |
| Extra SMEM | 0 | `s_u` (324 double) + `s_max` (256 double) | stesso di V2 |

### 7.2 Perché V2 è più veloce di V1

In V1 ogni punto interno genera 5 global memory reads (4 vicini + f). Nell'accesso allo stencil, i punti vicini di thread diversi si sovrappongono: il punto `u[i+1, j]` è richiesto sia dal thread `(i,j)` che dal thread `(i+2,j)`. In V2 questo dato è caricato una volta in shmem e condiviso.

**Risparmio stimato di bandwidth:**  
Per un tile 16×16 = 256 punti interni:
- V1: 256 × 5 = 1280 global reads per tile
- V2: (16+2)×(16+2) = 324 carichi shmem + 256 reads di `f` ≈ 580 global reads per tile
- **Risparmio ≈ 55% della bandwidth**

### 7.3 Perché V3 potrebbe essere ancora più veloce di V2

In V2, il caricamento dell'halo sinistro/destro usa thread di bordo con stride N (accessi verticali non coalescenti). In V3 tutti i 256 thread cooperano a caricare 324 elementi linearmente, massimizzando la coalescenza. Il beneficio è visibile specialmente per grandi N dove la bandwidth è il collo di bottiglia.

---

## 8. Flusso di Esecuzione Completo

L'esecuzione del programma segue un flusso deterministico diviso in quattro macro-fasi logiche: **Setup e Inizializzazione**, **Esecuzione dei Risolutori**, **Validazione della Correttezza Scientifica**, e **Benchmarking e Profilazione di Scalabilità**.

### 8.1 Schema del Flusso Logico

```
Program Start
│
├─ Parse CLI args (--n, --tol, --max-iter, --check-every, ...)
├─ Arrotonda N al multiplo di TILE_X se necessario
├─ Crea SolverParams {N, h, tol, max_iter, check_every}
│
├─ print_gpu_properties()     → Mostra nome GPU, memoria, compute capability
│
├─ test_standalone_reduction()
│     ├─ Genera vettore 2.5M elementi casuali
│     ├─ CPU max (loop seriale)
│     ├─ GPU max (reduce_max_gpu)
│     └─ Verifica |cpu-gpu| < 1e-9 → [PASS/FAIL]
│
├─ Alloca host: h_u_cpu, h_u_new, h_f, h_u_gpu_v1, h_u_gpu_v2, h_u_gpu_v3
├─ initialize(h_u_cpu, h_f, N, h)
├─ memcpy h_u_cpu → tutti i buffer GPU host
├─ Alloca device: d_u, d_u_new, d_f
│
├─ [CPU] jacobi_cpu(params, h_u_cpu)
│     └─ Loop: jacobi_step_cpu → swap → stop se error < tol
│
├─ [GPU V1] cudaMemcpy H→D | jacobi_gpu_naive(params, d_u, d_u_new, d_f)
│     └─ Loop: kernel_naive → swap | ogni 100 iter: copia N² double → max su CPU
│
├─ [GPU V2] cudaMemcpy H→D | jacobi_gpu_optimized(params, ...)
│     └─ Loop: kernel_shared → swap | ogni 100 iter: copia num_blocks double → max su CPU
│
├─ [GPU V3] cudaMemcpy H→D | jacobi_gpu_coalesced(params, ...)
│     └─ Loop: kernel_shared_coalesced → swap | ogni 100 iter: stessa di V2
│
├─ cudaMemcpy D→H (risultati finali con logica swap)
│
├─ VALIDAZIONE
│     ├─ print_validation(cpu, gpu_v1): max|diff|, RMS → [PASS/FAIL]
│     ├─ print_validation(cpu, gpu_v2)
│     ├─ print_validation(cpu, gpu_v3)
│     ├─ max_error_vs_exact(h_u_cpu)   → Errore vs soluzione analitica
│     ├─ max_error_vs_exact(h_u_gpu_v1)
│     ├─ max_error_vs_exact(h_u_gpu_v2)
│     └─ max_error_vs_exact(h_u_gpu_v3)
│
├─ cudaFree d_u, d_u_new, d_f
├─ free h_u_cpu, h_u_new, h_u_gpu_*
│
└─ BENCHMARK SUITE (se !no-scalability)
      Per N in {128, 256, 512, 1024}:
        ├─ initialize + cudaMemcpy H→D
        ├─ CPU: jacobi_step_cpu × N_iter (scalato per N grandi)
        ├─ GPU V1: jacobi_gpu_benchmark(version=0, 1000 iter)
        ├─ GPU V2: jacobi_gpu_benchmark(version=1, 1000 iter)
        ├─ GPU V3: jacobi_gpu_benchmark(version=2, 1000 iter)
        └─ Stampa: N | Solver | Time(ms) | MUpdates/s | GB/s | Speedup
```

### 8.2 Dettaglio Fasi di Esecuzione

#### Fase 1: Setup e Inizializzazione
- **Parsing CLI e Allineamento**: Viene letto l'input utente e allineata la coordinata $N$. I parametri vengono salvati nella struttura `SolverParams`.
- **Query Dispositivo**: Viene invocata `cudaGetDeviceProperties` per controllare la presenza di una GPU compatibile.
- **Unit Test Riduzione**: Viene avviato `test_standalone_reduction` su memoria separata per testare l'algoritmo parallelo di riduzione su 2.5 milioni di campioni.
- **Allocazione Host**: Vengono allocati in RAM i vettori per la CPU, per il termine sorgente $f$ e per contenere le soluzioni restituite dalle 3 varianti GPU.
- **Inizializzazione Griglia**: Tramite `initialize()`, il bordo viene impostato a 0, le celle interne a 0 e il termine sorgente $f$ viene popolato secondo la formula analitica.
- **Allocazione e Copia Device**: Vengono creati tre buffer in VRAM (`d_u`, `d_u_new`, `d_f`) tramite `cudaMalloc`. Il termine sorgente $f$ e lo stato iniziale $u$ vengono caricati sulla GPU tramite `cudaMemcpy(..., cudaMemcpyHostToDevice)`.

#### Fase 2: Cicli di Risoluzione
1. **CPU Solver**: Esegue lo sweep Jacobi in modo sequenziale su CPU misurando il tempo complessivo tramite `CpuTimer`.
2. **GPU V1 (Naive)**: 
   - I dati `d_u` e `d_u_new` sul device vengono resettati alla condizione iniziale.
   - Avvia il ciclo Jacobi invocando `jacobi_kernel_naive`.
   - Ogni 100 iterazioni (Strategy A), effettua una sincronizzazione implicita e scarica l'intera memoria di griglia sull'host per consentire alla CPU di valutare la convergenza.
3. **GPU V2 (Shared)**:
   - Resetta la memoria della GPU.
   - Avvia il ciclo Jacobi invocando `jacobi_kernel_shared`.
   - Ogni 100 iterazioni (Strategy B), legge l'array ridotto `d_block_max` copiando solo pochi kilobyte tramite PCIe. La CPU esegue la riduzione finale del piccolo vettore.
4. **GPU V3 (Coalesced)**:
   - Simile a V2, ma esegue l'aggiornamento invocando `jacobi_kernel_shared_coalesced` per velocizzare il caricamento della memoria condivisa.
5. **Copia Finale**: Identifica il buffer corretto contenente i dati definitivi in base al numero di iterazioni effettuate (pari o dispari) e li trasferisce in RAM.

#### Fase 3: Validazione
- Vengono eseguiti i confronti incrociati tra la CPU (baseline) e le 3 soluzioni GPU misurando la massima differenza assoluta (`max_abs_diff`) e l'errore quadratico medio (`rms_diff`).
- Ciascuna delle 4 soluzioni numeriche viene confrontata con la soluzione matematica analitica $u(x,y) = \sin(\pi x)\sin(\pi y)$ tramite `max_error_vs_exact`.

#### Fase 4: Benchmark Suite
- Libera tutte le risorse allocate nella fase precedente tramite `cudaFree` e `free`.
- Per ciascuna dimensione di griglia $N \in \{128, 256, 512, 1024\}$:
  - Alloca e inizializza i dati su host e device.
  - Esegue 1000 iterazioni fisse di ciascun solutore per escludere l'overhead del convergence check.
  - Per ciascun test GPU viene inserito un passo di Warmup iniziale prima di avviare il timer.
  - Stampa la tabella riassuntiva delle prestazioni in termini di MUpdates/s, larghezza di banda (GB/s) e fattore di Speedup.
  - Libera la memoria temporanea.


---