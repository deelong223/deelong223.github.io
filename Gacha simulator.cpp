#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

using namespace std;

// Constants
const double ROLE_WRONG_PROB = 0.49982;   // Probability of getting wrong character
const double ROLE_RIGHT_PROB = 1.0 - ROLE_WRONG_PROB; // Probability of getting target character
const double WEAPON_A_PROB = 0.375;       // Probability of getting weapon a (non-guaranteed)
const double WEAPON_B_PROB = 0.375;       // Probability of getting weapon b
const double WEAPON_C_PROB = 0.25;        // Probability of getting junk weapon

// Character pool state encoding (last, cnt) total 7 states
// last: 0=got target, 1=got wrong
// cnt: number of consecutive wrong-target cycles completed
// State indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2)
const int role_state_last[7] = { 0,0,0,0,1,1,1 };
const int role_state_cnt[7] = { 0,1,2,3,0,1,2 };

// Get state index from last and cnt
int get_role_state(int last, int cnt) {
    for (int i = 0; i < 7; ++i) {
        if (role_state_last[i] == last && role_state_cnt[i] == cnt)
            return i;
    }
    return -1; // Should never happen
}

// Precomputed character pool probability distribution
vector<vector<double>> role_f; // role_f[y][x] probability of getting 5-star at x-th draw starting from y draws since last 5-star
void init_role_f() {
    // Calculate conditional probability p[t] for each draw (t from 1 to 90)
    vector<double> p(91, 0.0);
    for (int t = 1; t <= 90; ++t) {
        if (t <= 73) p[t] = 0.006;
        else if (t <= 89) p[t] = 0.006 + (t - 73) * 0.06;
        else p[t] = 1.0;
    }
    // Calculate role_f[y][x]
    role_f.assign(90, vector<double>(91, 0.0)); // y from 0 to 89, x from 1 to 90-y
    for (int y = 0; y < 90; ++y) {
        double q = 1.0; // Probability of not getting 5-star in previous x-1 draws
        for (int x = 1; x <= 90 - y; ++x) {
            int t = y + x;
            role_f[y][x] = q * p[t];
            q *= (1.0 - p[t]);
        }
    }
}

// Precomputed weapon pool probability distribution
vector<vector<double>> weapon_f; // weapon_f[y][x] probability of getting 5-star at x-th draw starting from y draws since last 5-star
void init_weapon_f() {
    vector<double> p(81, 0.0);
    for (int t = 1; t <= 80; ++t) {
        if (t <= 62) p[t] = 0.007;
        else if (t <= 73) p[t] = 0.007 + (t - 62) * 0.07;
        else if (t <= 79) p[t] = 0.007 + 11 * 0.07 + (t - 73) * 0.035; // 73rd draw: 0.007+0.77=0.777
        else p[t] = 1.0;
    }
    weapon_f.assign(80, vector<double>(81, 0.0)); // y from 0 to 79, x from 1 to 80-y
    for (int y = 0; y < 80; ++y) {
        double q = 1.0;
        for (int x = 1; x <= 80 - y; ++x) {
            int t = y + x;
            weapon_f[y][x] = q * p[t];
            q *= (1.0 - p[t]);
        }
    }
}

// Character pool recursive distribution calculation
vector<vector<vector<double>>> role_memo; // role_memo[state][k] stores distribution starting from state, y=0, need k more targets
int role_k_max; // Maximum k

vector<double> role_dist(int state, int k, int y) {
    if (k == 0) {
        return vector<double>(1, 1.0); // Only t=0 has probability 1
    }
    if (y == 0 && !role_memo[state][k].empty()) {
        return role_memo[state][k];
    }
    int max_t = 90 * 2 * k; // Theoretical maximum draws
    vector<double> res(max_t + 1, 0.0);
    int max_x = (y == 0) ? 90 : (90 - y);
    int last = role_state_last[state];
    int cnt = role_state_cnt[state];

    // Generate possible outcomes
    vector<pair<int, int>> outcomes; // (new_state, new_k)
    vector<double> probs;

    if (last == 1) { // Last was wrong, next is guaranteed target
        int new_state = get_role_state(0, cnt + 1);
        outcomes.push_back({ new_state, k - 1 }); // Got target, k-1
        probs.push_back(1.0);
    }
    else { // last == 0
        if (cnt < 3) {
            // Wrong branch
            int new_state_w = get_role_state(1, cnt);
            outcomes.push_back({ new_state_w, k }); // Got wrong, k unchanged
            probs.push_back(ROLE_WRONG_PROB);
            // Target branch
            int new_state_z = get_role_state(0, 0);
            outcomes.push_back({ new_state_z, k - 1 }); // Got target
            probs.push_back(ROLE_RIGHT_PROB);
        }
        else { // cnt == 3, must get target
            int new_state = get_role_state(0, 0);
            outcomes.push_back({ new_state, k - 1 });
            probs.push_back(1.0);
        }
    }

    // Iterate over all possible x
    for (int x = 1; x <= max_x; ++x) {
        double prob_x = role_f[y][x];
        for (size_t idx = 0; idx < outcomes.size(); ++idx) {
            int new_state = outcomes[idx].first;
            int new_k = outcomes[idx].second;
            double prob_r = probs[idx];
            vector<double> sub = role_dist(new_state, new_k, 0);
            // Add shifted sub-distribution to res
            for (size_t t = 0; t < sub.size(); ++t) {
                if (sub[t] != 0.0) {
                    res[x + t] += prob_x * prob_r * sub[t];
                }
            }
        }
    }

    // If y==0, store in memoization
    if (y == 0) {
        role_memo[state][k] = res;
    }
    return res;
}

// Weapon pool recursive distribution calculation
vector<vector<vector<vector<double>>>> weapon_memo; // weapon_memo[i][j][flag] stores distribution starting from y=0
int weapon_i_max, weapon_j_max;

vector<double> weapon_dist(int i, int j, int flag, int y) {
    if (i == 0 && j == 0) {
        return vector<double>(1, 1.0);
    }
    if (y == 0 && !weapon_memo[i][j][flag].empty()) {
        return weapon_memo[i][j][flag];
    }
    int max_t = 80 * 2 * (i + j); // Theoretical maximum
    vector<double> res(max_t + 1, 0.0);
    int max_x = (y == 0) ? 80 : (80 - y);

    // Determine current target
    int spec = (i > 0) ? 0 : 1; // 0 means targeting a, 1 means targeting b

    // Outcome branches
    vector<int> new_i, new_j, new_flag;
    vector<double> probs;

    if (flag == 1) { // Guaranteed, will get target
        if (spec == 0) {
            new_i.push_back(i - 1);
            new_j.push_back(j);
            new_flag.push_back(0);
            probs.push_back(1.0);
        }
        else {
            new_i.push_back(i);
            new_j.push_back(j - 1);
            new_flag.push_back(0);
            probs.push_back(1.0);
        }
    }
    else { // No guarantee
        // Branch for getting target A
        if (spec == 0) {
            new_i.push_back(i - 1);
            new_j.push_back(j);
            new_flag.push_back(0);
            probs.push_back(WEAPON_A_PROB);
        }
        else {
            new_i.push_back(i);
            new_j.push_back(j - 1);
            new_flag.push_back(0);
            probs.push_back(WEAPON_B_PROB);
        }

        // Branch for getting the other weapon (non-target, but still a 5-star)
        if (spec == 0) {
            // Getting B when targeting A
            new_i.push_back(i);          // A需求不变
            new_j.push_back(j > 0 ? j - 1 : 0); // 如果还需要B，则减少；否则不变
            new_flag.push_back(1);        // 触发保底
            probs.push_back(WEAPON_B_PROB);
        }
        else {
            // Getting A when targeting B
            new_i.push_back(i > 0 ? i - 1 : 0);
            new_j.push_back(j);
            new_flag.push_back(1);
            probs.push_back(WEAPON_A_PROB);
        }

        // Branch for getting junk
        new_i.push_back(i);
        new_j.push_back(j);
        new_flag.push_back(1);
        probs.push_back(WEAPON_C_PROB);
    }

    // Iterate over x
    for (int x = 1; x <= max_x; ++x) {
        double prob_x = weapon_f[y][x];
        for (size_t idx = 0; idx < probs.size(); ++idx) {
            int ni = new_i[idx], nj = new_j[idx], nf = new_flag[idx];
            double pr = probs[idx];
            // 确保需求不会变成负数（上面已经处理，这里作为安全检查）
            if (ni < 0 || nj < 0) continue;
            vector<double> sub = weapon_dist(ni, nj, nf, 0);
            for (size_t t = 0; t < sub.size(); ++t) {
                if (sub[t] != 0.0) {
                    res[x + t] += prob_x * pr * sub[t];
                }
            }
        }
    }

    if (y == 0) {
        weapon_memo[i][j][flag] = res;
    }
    return res;
}

// Function to display menu and get input
void displayMenu() {
    cout << "\n=========================================" << endl;
    cout << "     GACHA PROBABILITY CALCULATOR" << endl;
    cout << "=========================================" << endl;
    cout << "\nThis calculator computes the probability of achieving" << endl;
    cout << "your desired pulls within a given budget." << endl;
    cout << "\nCharacter Pool Rules:" << endl;
    cout << "- Base 5-star rate: 0.6%" << endl;
    cout << "- Soft pity starts at 74th draw (+6% per draw)" << endl;
    cout << "- Hard pity at 90th draw (100%)" << endl;
    cout << "- When getting 5-star: 49.982% chance to get wrong character" << endl;
    cout << "- After getting wrong character, next 5-star is guaranteed target" << endl;
    cout << "- Cannot have more than 3 consecutive wrong-target cycles" << endl;
    cout << "\nWeapon Pool Rules:" << endl;
    cout << "- Base 5-star rate: 0.7%" << endl;
    cout << "- Soft pity starts at 63rd draw (+7% per draw until 73)" << endl;
    cout << "- Then +3.5% per draw until 79" << endl;
    cout << "- Hard pity at 80th draw (100%)" << endl;
    cout << "- Can choose between weapon A and B as target" << endl;
    cout << "- When getting 5-star: 37.5% A, 37.5% B, 25% junk" << endl;
    cout << "- After getting junk, next 5-star is guaranteed target" << endl;
    cout << "- Weapon pool guarantee does NOT carry over to next banner" << endl;
    cout << "=========================================\n" << endl;
}

int main() {
    // Initialize probability tables
    init_role_f();
    init_weapon_f();

    displayMenu();

    // Get input with prompts
    cout << "=== CHARACTER POOL STATUS ===" << endl;
    cout << "Draws since last 5-star (0-89): ";
    int y_r;
    cin >> y_r;

    cout << "Last 5-star result (0=target, 1=wrong): ";
    int last_r;
    cin >> last_r;

    cout << "Number of completed wrong-target cycles (0-3): ";
    int cycle_r;
    cin >> cycle_r;

    cout << "\n=== WEAPON POOL STATUS ===" << endl;
    cout << "Draws since last 5-star (0-79): ";
    int y_w;
    cin >> y_w;

    cout << "Last 5-star result (1=target/junk, 0=guarantee next): ";
    int last_w;
    cin >> last_w;

    cout << "\n=== TARGETS AND BUDGET ===" << endl;
    cout << "Number of target characters needed: ";
    int role_target;
    cin >> role_target;

    cout << "Number of weapon A needed: ";
    int a_target;
    cin >> a_target;

    cout << "Number of weapon B needed: ";
    int b_target;
    cin >> b_target;

    cout << "Total budget (number of pulls): ";
    int budget;
    cin >> budget;

    // Determine initial character pool state
    int role_state;
    if (last_r == 1) {
        role_state = get_role_state(1, cycle_r);
    }
    else {
        role_state = get_role_state(0, cycle_r);
    }

    // Determine initial weapon pool flag
    int weapon_flag = last_w; // 1 means have guarantee

    // Allocate memoization space
    role_k_max = role_target;
    role_memo.assign(7, vector<vector<double>>(role_k_max + 1));
    weapon_i_max = a_target;
    weapon_j_max = b_target;
    weapon_memo.assign(weapon_i_max + 1,
        vector<vector<vector<double>>>(weapon_j_max + 1,
            vector<vector<double>>(2)));

    cout << "\nCalculating probabilities... (this may take a moment)" << endl;

    // Calculate distributions
    vector<double> role_distribution = role_dist(role_state, role_target, y_r);
    vector<double> weapon_distribution = weapon_dist(a_target, b_target, weapon_flag, y_w);

    // Truncate to budget
    int max_len = budget + 1;
    if (role_distribution.size() > max_len) role_distribution.resize(max_len);
    if (weapon_distribution.size() > max_len) weapon_distribution.resize(max_len);
    role_distribution.resize(max_len, 0.0);
    weapon_distribution.resize(max_len, 0.0);

    // Calculate weapon pool cumulative distribution
    vector<double> weapon_cum(max_len + 1, 0.0);
    for (int s = 0; s <= budget; ++s) {
        weapon_cum[s + 1] = weapon_cum[s] + weapon_distribution[s];
    }

    // Calculate total probability
    double total_prob = 0.0;
    for (int r = 0; r <= budget; ++r) {
        total_prob += role_distribution[r] * weapon_cum[budget - r + 1];
    }

    // Output results
    cout << "\n=========================================" << endl;
    cout << "              RESULTS" << endl;
    cout << "=========================================" << endl;
    cout << "Weapon pool guarantee does NOT carry over to next banner" << endl;
    cout << "\nInput Summary:" << endl;
    cout << "- Character pool: " << y_r << " draws since last, last=" << (last_r ? "wrong" : "target")
        << ", cycles=" << cycle_r << endl;
    cout << "- Weapon pool: " << y_w << " draws since last, last=" << (last_w ? "guarantee next" : "no guarantee") << endl;
    cout << "- Targets: " << role_target << " characters, " << a_target << " weapon A, " << b_target << " weapon B" << endl;
    cout << "- Budget: " << budget << " pulls" << endl;
    cout << "\nProbability of achieving all targets within budget:" << endl;
    cout << fixed << setprecision(6) << total_prob * 100 << "%" << endl;
    cout << "=========================================" << endl;

    return 0;
}