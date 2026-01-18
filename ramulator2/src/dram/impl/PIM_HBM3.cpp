#include "dram/dram.h"
#include "dram/lambdas.h"

namespace Ramulator {

class PIM_HBM3 : public IDRAM, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAM, PIM_HBM3, "PIM_HBM3", "PIM-Enabled HBM3 Device Model")

  public:
    inline static const std::map<std::string, Organization> org_presets = {
      //   name     density   DQ    Ch Pch  Bg Ba   Ro     Co
      {"HBM3_8Gb",   {8<<10,  128,  {1, 2,  4,  4, 1<<15, 1<<6}}},
      {"PIM_8Gb",    {8<<10,  128,  {1, 2,  4,  4, 1<<15, 1<<6}}}, // Same org
    };

    inline static const std::map<std::string, std::vector<int>> timing_presets = {
      //   name       rate   nBL  nCL  nRCDRD  nRCDWR  nRP  nRAS  nRC  nWR  nRTPS  nRTPL  nCWL  nCCDS  nCCDL  nRRDS  nRRDL  nWTRS  nWTRL  nRTW  nFAW  nRFC  nRFCSB  nREFI  nREFISB  nRREFD  tCK_ps  nMAC
      {"HBM3_2Gbps_PIM",  {2000,   4,   7,    7,      7,     7,   17,  19,   8,    2,     3,    2,    1,      2,     2,     3,     3,     4,    3,    15,   -1,   160,   3900,     -1,      8,   1000,  16}},
    };

  /************************************************
   *                Organization
   ***********************************************/   
    const int m_internal_prefetch_size = 2;
    inline static constexpr ImplDef m_levels = {
      "channel", "pseudochannel", "bankgroup", "bank", "row", "column",    
    };

  /************************************************
   *             Requests & Commands
   ***********************************************/
    inline static constexpr ImplDef m_commands = {
      "ACT", 
      "PRE", "PREA",
      "RD",  "WR",  "RDA",  "WRA",
      "REFab", "REFsb",
      "RFMab", "RFMsb",
      "PIM_MAC" // <--- NEW
    };

    inline static const ImplLUT m_command_scopes = LUT (
      m_commands, m_levels, {
        {"ACT",   "row"},
        {"PRE",   "bank"},    {"PREA",   "channel"},
        {"RD",    "column"},  {"WR",     "column"}, {"RDA",   "column"}, {"WRA",   "column"},
        {"REFab", "channel"}, {"REFsb",  "bank"},
        {"RFMab", "channel"}, {"RFMsb",  "bank"},
        {"PIM_MAC", "column"}, // PIM operates on open row data
      }
    );

    inline static const ImplLUT m_command_meta = LUT<DRAMCommandMeta> (
      m_commands, {
                // open?   close?   access?  refresh?
        {"ACT",   {true,   false,   false,   false}},
        {"PRE",   {false,  true,    false,   false}},
        {"PREA",  {false,  true,    false,   false}},
        {"RD",    {false,  false,   true,    false}},
        {"WR",    {false,  false,   true,    false}},
        {"RDA",   {false,  true,    true,    false}},
        {"WRA",   {false,  true,    true,    false}},
        {"REFab", {false,  false,   false,   true }},
        {"REFsb", {false,  false,   false,   true }},
        {"RFMab", {false,  false,   false,   true }},
        {"RFMsb", {false,  false,   false,   true }},
        {"PIM_MAC", {false, false,  true,    false}},
      }
    );

    inline static constexpr ImplDef m_requests = {
      "read", "write", "all-bank-refresh", "per-bank-refresh", "all-bank-rfm", "per-bank-rfm", "pim-mac"
    };

    inline static const ImplLUT m_request_translations = LUT (
      m_requests, m_commands, {
        {"read", "RD"}, {"write", "WR"}, {"all-bank-refresh", "REFab"}, {"per-bank-refresh", "REFsb"}, 
        {"all-bank-rfm", "RFMab"}, {"per-bank-rfm", "RFMsb"},
        {"pim-mac", "PIM_MAC"} 
      }
    );

   
  /************************************************
   *                   Timing
   ***********************************************/
    inline static constexpr ImplDef m_timings = {
      "rate", 
      "nBL", "nCL", "nRCDRD", "nRCDWR", "nRP", "nRAS", "nRC", "nWR", "nRTPS", "nRTPL", "nCWL",
      "nCCDS", "nCCDL",
      "nRRDS", "nRRDL",
      "nWTRS", "nWTRL",
      "nRTW",
      "nFAW",
      "nRFC", "nRFCSB", "nREFI", "nREFISB", "nRREFD",
      "tCK_ps",
      "nMAC" // <--- NEW
    };

  /************************************************
   *                 Node States
   ***********************************************/
    inline static constexpr ImplDef m_states = {
       "Opened", "Closed", "N/A", "Refreshing"
    };

    inline static const ImplLUT m_init_states = LUT (
      m_levels, m_states, {
        {"channel",       "N/A"}, 
        {"pseudochannel", "N/A"}, 
        {"bankgroup",     "N/A"},
        {"bank",          "Closed"},
        {"row",           "Closed"},
        {"column",        "N/A"},
      }
    );

  public:
    struct Node : public DRAMNodeBase<PIM_HBM3> {
      Node(PIM_HBM3* dram, Node* parent, int level, int id) : DRAMNodeBase<PIM_HBM3>(dram, parent, level, id) {};
    };
    std::vector<Node*> m_channels;
    
    FuncMatrix<ActionFunc_t<Node>>  m_actions;
    FuncMatrix<PreqFunc_t<Node>>    m_preqs;
    FuncMatrix<RowhitFunc_t<Node>>  m_rowhits;
    FuncMatrix<RowopenFunc_t<Node>> m_rowopens;


  public:
    void tick() override {
      m_clk++;
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
      m_channels[channel_id]->update_states(command, addr_vec, m_clk);
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
      // Channel width
      m_channel_width = param_group("org").param<int>("channel_width").default_val(64);
      m_organization.count.resize(m_levels.size(), -1);

      if (auto preset_name = param_group("org").param<std::string>("preset").optional()) {
        if (org_presets.count(*preset_name) > 0) {
          m_organization = org_presets.at(*preset_name);
        }
      }
      // Simplified org loading...
    };

    void set_timing_vals() {
      m_timing_vals.resize(m_timings.size(), -1);
      if (auto preset_name = param_group("timing").param<std::string>("preset").optional()) {
        m_timing_vals = timing_presets.at(*preset_name);
      }
      int tCK_ps = m_timing_vals("tCK_ps");
      m_read_latency = m_timing_vals("nCL") + m_timing_vals("nBL");

      #define V(timing) (m_timing_vals(timing))
      populate_timingcons(this, {
          /*** Channel ***/ 
          {.level = "channel", .preceding = {"ACT"}, .following = {"ACT", "PRE", "PREA", "REFab", "REFsb", "RFMab", "RFMsb"}, .latency = 2},

          /*** Pseudo Channel ***/ 
          // Standard HBM3 Constraints
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDS")},
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nFAW"), .window = 4},
          {.level = "pseudochannel", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nBL")},
          {.level = "pseudochannel", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nBL")},
          
          // PIM Specific Constraints
          // 1. PIM needs data from Open Row (ACT -> PIM)
          {.level = "bank", .preceding = {"ACT"}, .following = {"PIM_MAC"}, .latency = V("nRCDRD")}, // Assume Read-like latency to start
          
          // 2. PIM Execution Time
          {.level = "bank", .preceding = {"PIM_MAC"}, .following = {"PRE"}, .latency = V("nMAC") + V("nRTPS")}, // Wait MAC finish
          
          // 3. PIM to PIM (Pipeline or Serial?) -> Let's assume Serial per bank
          {.level = "bank", .preceding = {"PIM_MAC"}, .following = {"PIM_MAC"}, .latency = V("nMAC")},

          // 4. PIM to Read/Write (Result retrieval)
          {.level = "bank", .preceding = {"PIM_MAC"}, .following = {"RD", "WR"}, .latency = V("nMAC")},
      });
      #undef V

    };

    void set_actions() {
      m_actions.resize(m_levels.size(), std::vector<ActionFunc_t<Node>>(m_commands.size()));
      // Channel Actions
      m_actions[m_levels["channel"]][m_commands["PREA"]] = Lambdas::Action::Channel::PREab<PIM_HBM3>;
      // Bank actions
      m_actions[m_levels["bank"]][m_commands["ACT"]] = Lambdas::Action::Bank::ACT<PIM_HBM3>;
      m_actions[m_levels["bank"]][m_commands["PRE"]] = Lambdas::Action::Bank::PRE<PIM_HBM3>;
      
      // PIM Action (No state change, just access)
      m_actions[m_levels["bank"]][m_commands["PIM_MAC"]] = [](Node* node, int cmd, int target_id, int clk) {}; 
    };

    void set_preqs() {
      m_preqs.resize(m_levels.size(), std::vector<PreqFunc_t<Node>>(m_commands.size()));
      m_preqs[m_levels["channel"]][m_commands["REFab"]] = Lambdas::Preq::Channel::RequireAllBanksClosed<PIM_HBM3>;
      m_preqs[m_levels["bank"]][m_commands["REFsb"]] = Lambdas::Preq::Bank::RequireBankClosed<PIM_HBM3>;
      m_preqs[m_levels["bank"]][m_commands["RD"]] = Lambdas::Preq::Bank::RequireRowOpen<PIM_HBM3>;
      m_preqs[m_levels["bank"]][m_commands["WR"]] = Lambdas::Preq::Bank::RequireRowOpen<PIM_HBM3>;
      
      // PIM Preq
      m_preqs[m_levels["bank"]][m_commands["PIM_MAC"]] = Lambdas::Preq::Bank::RequireRowOpen<PIM_HBM3>;
    };

    void set_rowhits() {
      m_rowhits.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));
      m_rowhits[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowHit::Bank::RDWR<PIM_HBM3>;
      m_rowhits[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowHit::Bank::RDWR<PIM_HBM3>;
      m_rowhits[m_levels["bank"]][m_commands["PIM_MAC"]] = Lambdas::RowHit::Bank::RDWR<PIM_HBM3>;
    }

    void set_rowopens() {
      m_rowopens.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));
      m_rowopens[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowOpen::Bank::RDWR<PIM_HBM3>;
      m_rowopens[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowOpen::Bank::RDWR<PIM_HBM3>;
      m_rowopens[m_levels["bank"]][m_commands["PIM_MAC"]] = Lambdas::RowOpen::Bank::RDWR<PIM_HBM3>;
    }

    void create_nodes() {
      int num_channels = m_organization.count[m_levels["channel"]];
      for (int i = 0; i < num_channels; i++) {
        m_channels.push_back(new Node(this, nullptr, 0, i));
      }
    };
};

} // namespace Ramulator
