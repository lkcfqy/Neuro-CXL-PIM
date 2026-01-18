#include "dram/dram.h"
#include "dram/lambdas.h"

namespace Ramulator {

class PIM_DDR4 : public IDRAM, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAM, PIM_DDR4, "PIM_DDR4", "PIM-Enabled DDR4 Device Model")

  public:
    inline static const std::map<std::string, Organization> org_presets = {
      {"DDR4_8Gb_x8",   {8<<10,   8,  {1, 1, 4, 4, 1<<16, 1<<10}}},
      {"PIM_8Gb_x8",    {8<<10,   8,  {1, 1, 4, 4, 1<<16, 1<<10}}}, // Same org as DDR4
    };

    inline static const std::map<std::string, std::vector<int>> timing_presets = {
      // Standard DDR4-2400 timings + nMAC (default 10 cycles for simple MAC)
      // Name             rate   nBL nCL nRCD nRP nRAS nRC nWR nRTP nCWL nCCDS nCCDL nRRDS nRRDL nWTRS nWTRL nFAW nRFC nREFI nCS tCK_ps  nMAC
      {"DDR4_2400R_PIM",  {2400,   4,  16,  16,  16,  39,  55, 18,  9,   12,   4,    6,    -1,   -1,   3,    9,    -1,  -1,   -1,  2,  833,    16}}, 
    };

    /************************************************
     *             Requests & Commands
     ***********************************************/
    inline static constexpr ImplDef m_commands = {
      "ACT", 
      "PRE", "PREA",
      "RD",  "WR",  "RDA",  "WRA",
      "REFab", "REFab_end",
      "PIM_MAC" // <--- NEW: PIM Multiply-Accumulate
    };

    inline static const ImplLUT m_command_scopes = LUT (
      m_commands, m_levels, {
        {"ACT",   "row"},
        {"PRE",   "bank"},   {"PREA",   "rank"},
        {"RD",    "column"}, {"WR",     "column"}, {"RDA",   "column"}, {"WRA",   "column"},
        {"REFab", "rank"},  {"REFab_end", "rank"},
        {"PIM_MAC", "column"} // PIM happens at column level (row buffer)
      }
    );

    inline static const ImplLUT m_command_meta = LUT<DRAMCommandMeta> (
      m_commands, {
                    // open?   close?   access?  refresh?
        {"ACT",       {true,   false,   false,   false}},
        {"PRE",       {false,  true,    false,   false}},
        {"PREA",      {false,  true,    false,   false}},
        {"RD",        {false,  false,   true,    false}},
        {"WR",        {false,  false,   true,    false}},
        {"RDA",       {false,  true,    true,    false}},
        {"WRA",       {false,  true,    true,    false}},
        {"REFab",     {false,  false,   false,   true }},
        {"REFab_end", {false,  true,    false,   false}},
        {"PIM_MAC",   {false,  false,   true,    false}}, // PIM is an access
      }
    );

    inline static constexpr ImplDef m_requests = {
      "read", "write", "all-bank-refresh", "open-row", "close-row", "pim-mac"
    };

    inline static const ImplLUT m_request_translations = LUT (
      m_requests, m_commands, {
        {"read", "RD"}, {"write", "WR"}, {"all-bank-refresh", "REFab"},
        {"open-row", "ACT"}, {"close-row", "PRE"},
        {"pim-mac", "PIM_MAC"}
      }
    );

    /************************************************
     *                   Timing
     ***********************************************/
    inline static constexpr ImplDef m_timings = {
      "rate", 
      "nBL", "nCL", "nRCD", "nRP", "nRAS", "nRC", "nWR", "nRTP", "nCWL",
      "nCCDS", "nCCDL",
      "nRRDS", "nRRDL",
      "nWTRS", "nWTRL",
      "nFAW",
      "nRFC","nREFI",
      "nCS",
      "tCK_ps",
      "nMAC" // <--- NEW PARAMETER
    };

    /************************************************
     *                   Organization (Standard)
     ***********************************************/
    const int m_internal_prefetch_size = 8;
    inline static constexpr ImplDef m_levels = {
      "channel", "rank", "bankgroup", "bank", "row", "column",    
    };

    /************************************************
     *                 Node States
     ***********************************************/
    inline static constexpr ImplDef m_states = {
       "Opened", "Closed", "PowerUp", "N/A", "Refreshing"
    };

    inline static const ImplLUT m_init_states = LUT (
      m_levels, m_states, {
        {"channel",   "N/A"}, 
        {"rank",      "PowerUp"},
        {"bankgroup", "N/A"},
        {"bank",      "Closed"},
        {"row",       "Closed"},
        {"column",    "N/A"},
      }
    );

    // For this prototype, we reuse standard voltage/current defs but they won't be used unless we implement set_powers fully.
    
    public:
    struct Node : public DRAMNodeBase<PIM_DDR4> {
      Node(PIM_DDR4* dram, Node* parent, int level, int id) : DRAMNodeBase<PIM_DDR4>(dram, parent, level, id) {};
    };
    std::vector<Node*> m_channels;
    
    // Function Matrices
    FuncMatrix<ActionFunc_t<Node>>  m_actions;
    FuncMatrix<PreqFunc_t<Node>>    m_preqs;
    FuncMatrix<RowhitFunc_t<Node>>  m_rowhits;
    FuncMatrix<RowopenFunc_t<Node>> m_rowopens;

    void tick() override {
      m_clk++;
      for (int i = m_future_actions.size() - 1; i >= 0; i--) {
        auto& future_action = m_future_actions[i];
        if (future_action.clk == m_clk) {
          handle_future_action(future_action.cmd, future_action.addr_vec);
          m_future_actions.erase(m_future_actions.begin() + i);
        }
      }
    };

    void init() override {
      RAMULATOR_DECLARE_SPECS();
      set_organization();
      set_timing_vals();

      set_actions();
      set_preqs();
      set_rowhits();
      set_rowopens();
      
      create_nodes();
    };

    void issue_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      m_channels[channel_id]->update_timing(command, addr_vec, m_clk);
      // Power updates would go here
      check_future_action(command, addr_vec);
    };

    void check_future_action(int command, const AddrVec_t& addr_vec) {
        if (command == m_commands("REFab")) {
            m_future_actions.push_back({command, addr_vec, m_clk + m_timing_vals("nRFC") - 1});
        }
    }

    void handle_future_action(int command, const AddrVec_t& addr_vec) {
        // Handle REFab_end etc.
    };

    int get_preq_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->get_preq_command(command, addr_vec, m_clk);
    };

    bool check_ready(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_ready(command, addr_vec, m_clk);
    };

    bool check_rowbuffer_hit(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_rowbuffer_hit(command, addr_vec, m_clk);
    };
    
    bool check_node_open(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_node_open(command, addr_vec, m_clk);
    };

  private:
    void set_organization() {
      m_channel_width = param_group("org").param<int>("channel_width").default_val(64);
      m_organization.count.resize(m_levels.size(), -1);
      if (auto preset_name = param_group("org").param<std::string>("preset").optional()) {
         m_organization = org_presets.at(*preset_name);
      }
      // ... (Rest of org logic simplified for brevity)
    };

    void set_timing_vals() {
      m_timing_vals.resize(m_timings.size(), -1);
      if (auto preset_name = param_group("timing").param<std::string>("preset").optional()) {
        m_timing_vals = timing_presets.at(*preset_name);
      }
      int tCK_ps = m_timing_vals("tCK_ps");
      m_read_latency = m_timing_vals("nCL") + m_timing_vals("nBL");

      // Populate Standard Timing Constraints
      #define V(timing) (m_timing_vals(timing))
      populate_timingcons(this, {
          // Same as DDR4 for standard commands
          {.level = "channel", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nBL")},
          {.level = "channel", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nBL")},
          
          // PIM Specific Constraints
          // PIM MAC behaves like a Write followed by internal processing
          {.level = "bank", .preceding = {"ACT"}, .following = {"PIM_MAC"}, .latency = V("nRCD")},
          {.level = "bank", .preceding = {"PIM_MAC"}, .following = {"PRE"}, .latency = V("nMAC") + V("nWR")}, // Must wait MAC to finish
          {.level = "bank", .preceding = {"PIM_MAC"}, .following = {"RD"},  .latency = V("nMAC")},
          {.level = "bank", .preceding = {"PIM_MAC"}, .following = {"WR"},  .latency = V("nMAC")},
          {.level = "bank", .preceding = {"PIM_MAC"}, .following = {"PIM_MAC"}, .latency = V("nMAC")}, // Back to back PIM
      });
      #undef V
    };

    void set_actions() {
      m_actions.resize(m_levels.size(), std::vector<ActionFunc_t<Node>>(m_commands.size()));
      // Rank
      m_actions[m_levels["rank"]][m_commands["PREA"]] = Lambdas::Action::Rank::PREab<PIM_DDR4>;
      m_actions[m_levels["rank"]][m_commands["REFab"]] = Lambdas::Action::Rank::REFab<PIM_DDR4>;
      // Bank
      m_actions[m_levels["bank"]][m_commands["ACT"]] = Lambdas::Action::Bank::ACT<PIM_DDR4>;
      m_actions[m_levels["bank"]][m_commands["PRE"]] = Lambdas::Action::Bank::PRE<PIM_DDR4>;
      
      // PIM_MAC does not close row, behaves like RD/WR regarding state
      m_actions[m_levels["bank"]][m_commands["PIM_MAC"]] = [](Node* node, int cmd, int target_id, int clk) {}; 
    };

    void set_preqs() {
      m_preqs.resize(m_levels.size(), std::vector<PreqFunc_t<Node>>(m_commands.size()));
      // Standard
      m_preqs[m_levels["rank"]][m_commands["REFab"]] = Lambdas::Preq::Rank::RequireAllBanksClosed<PIM_DDR4>;
      m_preqs[m_levels["bank"]][m_commands["RD"]] = Lambdas::Preq::Bank::RequireRowOpen<PIM_DDR4>;
      m_preqs[m_levels["bank"]][m_commands["WR"]] = Lambdas::Preq::Bank::RequireRowOpen<PIM_DDR4>;
      m_preqs[m_levels["bank"]][m_commands["ACT"]] = Lambdas::Preq::Bank::RequireRowOpen<PIM_DDR4>;
      m_preqs[m_levels["bank"]][m_commands["PRE"]] = Lambdas::Preq::Bank::RequireBankClosed<PIM_DDR4>;

      // PIM Requirements: Row Must be Open
      m_preqs[m_levels["bank"]][m_commands["PIM_MAC"]] = Lambdas::Preq::Bank::RequireRowOpen<PIM_DDR4>;
    };

    void set_rowhits() {
      m_rowhits.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));
      m_rowhits[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowHit::Bank::RDWR<PIM_DDR4>;
      m_rowhits[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowHit::Bank::RDWR<PIM_DDR4>;
      
      // PIM is also a row hit if the row is open
      m_rowhits[m_levels["bank"]][m_commands["PIM_MAC"]] = Lambdas::RowHit::Bank::RDWR<PIM_DDR4>;
    }

    void set_rowopens() {
      m_rowopens.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));
      m_rowopens[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowOpen::Bank::RDWR<PIM_DDR4>;
      m_rowopens[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowOpen::Bank::RDWR<PIM_DDR4>;
      m_rowopens[m_levels["bank"]][m_commands["PIM_MAC"]] = Lambdas::RowOpen::Bank::RDWR<PIM_DDR4>;
    }

    void create_nodes() {
      int num_channels = m_organization.count[m_levels["channel"]];
      for (int i = 0; i < num_channels; i++) {
        m_channels.push_back(new Node(this, nullptr, 0, i));
      }
    }
};

} // namespace Ramulator
