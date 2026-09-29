(function () {
  "use strict";

  var requestedStage = new URLSearchParams(window.location.search).get("stage");
  var initialStage = ["serial", "scoreboard", "static", "compiler"].indexOf(requestedStage) !== -1
    ? requestedStage
    : "serial";
  var state = { stage: initialStage, cycle: 0, timer: null, inspectId: null };

  function instruction(id, asm, src, dst, unit, latency, control, options) {
    var settings = options || {};
    return {
      id: id,
      asm: asm,
      src: src,
      dst: dst,
      unit: unit,
      latency: latency,
      variable: Boolean(settings.variable),
      readDelay: settings.readDelay || 1,
      control: control
    };
  }

  var baseProgram = [
    instruction("I1", "FADD R1, R2, R3", ["R2", "R3"], "R1", "FP", 4,
      { stall: 0, wb: null, rd: null, wait: [] }),
    instruction("I2", "IADD R4, R5, R6", ["R5", "R6"], "R4", "INT", 1,
      { stall: 2, wb: null, rd: null, wait: [] }),
    instruction("I3", "FFMA R7, R1, R2, R3", ["R1", "R2", "R3"], "R7", "FP", 4,
      { stall: 0, wb: null, rd: null, wait: [] }),
    instruction("I4", "LOAD R1, [R2]", ["R2"], "R1", "MEM", 6,
      { stall: 0, wb: "SB0", rd: null, wait: [] }, { variable: true }),
    instruction("I5", "IADD R4, R5, R6", ["R5", "R6"], "R4", "INT", 1,
      { stall: 0, wb: null, rd: null, wait: [] }),
    instruction("I6", "IADD R7, R1, R8", ["R1", "R8"], "R7", "INT", 1,
      { stall: 0, wb: null, rd: null, wait: ["SB0"] }),
    instruction("I7", "LOAD R1, [R2]", ["R2"], "R1", "MEM", 5,
      { stall: 0, wb: "SB1", rd: null, wait: [] }, { variable: true }),
    instruction("I8", "IADD R1, R3, R4", ["R3", "R4"], "R1", "INT", 1,
      { stall: 0, wb: null, rd: null, wait: ["SB1"] }),
    instruction("I9", "LOAD R7, [R8]", ["R8"], "R7", "MEM", 7,
      { stall: 0, wb: null, rd: "SB2", wait: [] }, { variable: true, readDelay: 3 }),
    instruction("I10", "IADD R8, R2, R3", ["R2", "R3"], "R8", "INT", 1,
      { stall: 0, wb: null, rd: null, wait: ["SB2"] })
  ];

  function compileStaticProgram(program) {
    var compiled = [];
    var writerReady = {};
    var readerDone = {};
    var nopNumber = 1;
    program.forEach(function (item) {
      var earliest = compiled.length;
      item.src.forEach(function (register) {
        earliest = Math.max(earliest, writerReady[register] || 0);
      });
      earliest = Math.max(earliest, writerReady[item.dst] || 0, readerDone[item.dst] || 0);
      while (compiled.length < earliest) {
        compiled.push({
          id: "N" + nopNumber,
          asm: "NOP",
          src: [],
          dst: null,
          unit: null,
          latency: 0,
          readDelay: 0,
          nop: true,
          control: { stall: 0, wb: null, rd: null, wait: [] }
        });
        nopNumber += 1;
      }
      var issueCycle = compiled.length;
      compiled.push(item);
      writerReady[item.dst] = issueCycle + item.latency;
      item.src.forEach(function (register) {
        readerDone[register] = Math.max(readerDone[register] || 0, issueCycle + item.readDelay);
      });
    });
    return compiled;
  }

  var staticProgram = compileStaticProgram(baseProgram);
  var stageModels = {
    serial: {
      label: "STAGE 0 · STRICT SERIAL",
      description: "硬件只有全局 active 状态：上一条指令完成后，下一条才能发射。",
      stateGroup: "registers",
      usesActive: true,
      program: baseProgram,
      checks: function (item) { return ["ACTIVE", item.unit]; }
    },
    scoreboard: {
      label: "STAGE 1 · HARDWARE SCOREBOARD",
      description: "每个寄存器把 W（未完成写）和 R（未完成读）放在同一张状态卡中。",
      stateGroup: "scoreboard",
      usesActive: false,
      program: baseProgram,
      checks: function (item) {
        return item.src.map(function (register) { return "W:" + register; })
          .concat(["W:" + item.dst, "R:" + item.dst, item.unit]);
      }
    },
    static: {
      label: "STAGE 2 · STATIC SCHEDULING",
      description: "编译器按已知延迟插入 NOP；硬件不保存寄存器依赖状态。",
      stateGroup: "none",
      usesActive: false,
      program: staticProgram,
      checks: function (item) { return item.nop ? [] : [item.unit]; }
    },
    compiler: {
      label: "STAGE 3 · CONTROL BITS + COUNTERS",
      description: "寄存器仍保存数据；发射逻辑不查 per-register busy，而是查 S 和 WAIT 指定的 counters。",
      stateGroup: "counters",
      usesActive: false,
      program: baseProgram,
      checks: function (item) { return ["STALL"].concat(item.control.wait, [item.unit]); }
    }
  };

  function allRegisterNames() {
    var names = [];
    for (var number = 1; number <= 8; number += 1) {
      names.push("R" + number);
    }
    return names;
  }

  function emptyRegisterMap(value) {
    var map = {};
    allRegisterNames().forEach(function (register) { map[register] = value; });
    return map;
  }

  function emptyCounters() {
    return { SB0: 0, SB1: 0, SB2: 0, SB3: 0, SB4: 0, SB5: 0 };
  }

  function cloneResources(resources) {
    return {
      writers: Object.assign({}, resources.writers),
      units: Object.assign({}, resources.units),
      readers: Object.assign({}, resources.readers),
      counters: Object.assign({}, resources.counters),
      stall: resources.stall,
      active: resources.active
    };
  }

  function targetInUse(resources, target) {
    if (target === "ACTIVE") { return resources.active !== null; }
    if (target === "FP" || target === "INT" || target === "MEM") { return resources.units[target]; }
    if (target === "STALL") { return resources.stall > 0; }
    if (Object.prototype.hasOwnProperty.call(resources.counters, target)) {
      return resources.counters[target] > 0;
    }
    if (target.indexOf("W:") === 0) { return resources.writers[target.slice(2)] === true; }
    if (target.indexOf("R:") === 0) { return resources.readers[target.slice(2)] > 0; }
    return false;
  }

  function simulate(stageName) {
    var model = stageModels[stageName];
    var program = model.program;
    var resources = {
      writers: emptyRegisterMap(false),
      units: { FP: false, INT: false, MEM: false },
      readers: emptyRegisterMap(0),
      counters: emptyCounters(),
      stall: 0,
      active: null
    };
    var inflight = [];
    var counterChanges = [];
    var readerClears = [];
    var nextIndex = 0;
    var cycles = [];
    var timeline = {};
    program.forEach(function (item) { timeline[item.id] = []; });

    for (var cycle = 0; cycle < 96; cycle += 1) {
      counterChanges.filter(function (change) { return change.cycle === cycle; }).forEach(function (change) {
        resources.counters[change.counter] = Math.max(0, resources.counters[change.counter] + change.delta);
      });
      counterChanges = counterChanges.filter(function (change) { return change.cycle > cycle; });
      readerClears.filter(function (change) { return change.cycle === cycle; }).forEach(function (change) {
        resources.readers[change.register] = Math.max(0, resources.readers[change.register] - 1);
      });
      readerClears = readerClears.filter(function (change) { return change.cycle > cycle; });

      var completing = inflight.filter(function (entry) { return entry.doneCycle === cycle; });
      completing.forEach(function (entry) {
        resources.writers[entry.instruction.dst] = false;
        resources.units[entry.instruction.unit] = false;
        if (resources.active === entry.instruction.id) { resources.active = null; }
        if (stageName === "compiler" && entry.instruction.control.wb !== null) {
          resources.counters[entry.instruction.control.wb] = Math.max(
            0, resources.counters[entry.instruction.control.wb] - 1
          );
        }
      });
      inflight = inflight.filter(function (entry) { return entry.doneCycle > cycle; });

      var candidate = nextIndex < program.length ? program[nextIndex] : null;
      var checked = candidate === null ? [] : model.checks(candidate);
      var beforeIssue = cloneResources(resources);
      var failed = checked.filter(function (target) { return targetInUse(beforeIssue, target); });
      var decision = candidate === null ? "DONE" : (failed.length > 0 ? "STALL" : "ISSUE");

      program.forEach(function (item) {
        var completion = completing.some(function (entry) { return entry.instruction.id === item.id; });
        var running = inflight.find(function (entry) { return entry.instruction.id === item.id; });
        var status = "";
        if (completion) { status = "WB"; }
        else if (candidate !== null && candidate.id === item.id) {
          status = decision === "ISSUE" ? (candidate.nop ? "NP" : "IS") : "WAIT";
        }
        else if (running !== undefined) { status = "EX"; }
        timeline[item.id].push(status);
      });

      cycles.push({
        cycle: cycle,
        resources: beforeIssue,
        candidate: candidate,
        candidateIndex: nextIndex,
        checked: checked,
        failed: failed,
        decision: decision
      });

      var stallAtStart = resources.stall;
      if (decision === "ISSUE") {
        if (candidate.nop) {
          nextIndex += 1;
        }
        else {
          resources.writers[candidate.dst] = true;
          resources.units[candidate.unit] = true;
          if (stageName === "serial") { resources.active = candidate.id; }
          if (stageName === "scoreboard") {
            candidate.src.forEach(function (register) {
              resources.readers[register] += 1;
              readerClears.push({ cycle: cycle + candidate.readDelay, register: register });
            });
          }
          if (stageName === "compiler") {
            resources.stall = candidate.control.stall;
            if (candidate.control.wb !== null) {
              counterChanges.push({ cycle: cycle + 1, counter: candidate.control.wb, delta: 1 });
            }
            if (candidate.control.rd !== null) {
              counterChanges.push({ cycle: cycle + 1, counter: candidate.control.rd, delta: 1 });
              counterChanges.push({
                cycle: cycle + candidate.readDelay,
                counter: candidate.control.rd,
                delta: -1
              });
            }
          }
          inflight.push({ instruction: candidate, doneCycle: cycle + candidate.latency });
          nextIndex += 1;
        }
      }
      if (stallAtStart > 0) { resources.stall = Math.max(0, resources.stall - 1); }
      if (nextIndex >= program.length && inflight.length === 0) { break; }
    }
    return { cycles: cycles, timeline: timeline, maxCycle: cycles.length - 1 };
  }

  var replays = {
    serial: simulate("serial"),
    scoreboard: simulate("scoreboard"),
    static: simulate("static"),
    compiler: simulate("compiler")
  };

  function byId(id) { return document.getElementById(id); }
  function currentReplay() { return replays[state.stage]; }
  function currentSnapshot() { return currentReplay().cycles[state.cycle]; }
  function currentProgram() { return stageModels[state.stage].program; }
  function getInstruction(id) {
    return currentProgram().find(function (item) { return item.id === id; }) || null;
  }

  function inspectedInstruction(snapshot) { return getInstruction(state.inspectId) || snapshot.candidate; }
  function evaluateInstruction(item, snapshot) {
    if (item === null) { return { checked: [], failed: [] }; }
    var checked = stageModels[state.stage].checks(item);
    return {
      checked: checked,
      failed: checked.filter(function (target) { return targetInUse(snapshot.resources, target); })
    };
  }

  function visibleRegisters(snapshot) {
    return allRegisterNames();
  }

  function hardwareTile(text, status, title, checked) {
    var classes = ["hardware-tile", status];
    if (checked) { classes.push("checked"); }
    return "<span class=\"" + classes.join(" ") + "\" title=\"" + escapeHtml(title) +
      "\" aria-label=\"" + escapeHtml(title) + "\">" + escapeHtml(text) + "</span>";
  }

  function scoreCell(label, value, checked, title) {
    var classes = ["score-cell", value > 0 ? "active" : "available"];
    if (checked) { classes.push("checked"); }
    return "<span class=\"" + classes.join(" ") + "\" title=\"" + escapeHtml(title) + "\">" +
      "<b>" + label + "</b><em>" + value + "</em></span>";
  }

  function stateGroup(title, content, note) {
    return "<section class=\"state-group\"><div class=\"state-group-heading\">" +
      "<p class=\"group-label\">" + escapeHtml(title) + "</p><span>" + escapeHtml(note) +
      "</span></div><div class=\"state-group-content\">" + content + "</div></section>";
  }

  function renderHardware(snapshot) {
    var model = stageModels[state.stage];
    var evaluation = evaluateInstruction(inspectedInstruction(snapshot), snapshot);
    var registers = visibleRegisters(snapshot);
    var groups = [];

    if (model.stateGroup === "scoreboard") {
      var cards = registers.map(function (register) {
        var writerTarget = "W:" + register;
        var readerTarget = "R:" + register;
        var writerValue = snapshot.resources.writers[register] ? 1 : 0;
        var readerValue = Number(snapshot.resources.readers[register]);
        return "<div class=\"score-register-card\"><strong class=\"score-register-name\">" + register +
          "</strong><div class=\"score-register-values\">" +
          scoreCell("W", writerValue, evaluation.checked.indexOf(writerTarget) !== -1,
            writerTarget + "=" + writerValue) +
          scoreCell("R", readerValue, evaluation.checked.indexOf(readerTarget) !== -1,
            readerTarget + "=" + readerValue) + "</div></div>";
      }).join("");
      groups.push(stateGroup("Register scoreboard · 固定 8 个寄存器", cards,
        "W=尚未完成的写；R=尚未发生的读"));
    }
    else {
      var registerTiles = registers.map(function (register) {
        var active = snapshot.resources.writers[register];
        return hardwareTile(register, active ? "active" : "available",
          active ? register + " 正在等待写回" : register + " 当前无未完成写", false);
      }).join("");
      groups.push(stateGroup(
        state.stage === "compiler" ? "Register file · 存数据，不参与本阶段的 issue check" : "Register file",
        "<div class=\"register-bank\">" + registerTiles + "</div>",
        "R1-R8 持久显示"
      ));
      if (model.stateGroup === "none") {
        groups.push(stateGroup("Dependency state",
          hardwareTile("NONE", "available", "硬件不保存寄存器依赖状态", false),
          "安全间隔已经由编译器写进指令流"));
      }
      if (model.stateGroup === "counters") {
        var counters = ["STALL", "SB0", "SB1", "SB2", "SB3", "SB4", "SB5"].map(function (name) {
          var value = name === "STALL" ? snapshot.resources.stall : snapshot.resources.counters[name];
          return hardwareTile(name + ":" + value, value > 0 ? "active" : "available",
            name + "=" + value, evaluation.checked.indexOf(name) !== -1);
        }).join("");
        groups.push(stateGroup("Issue state · control bits 只读取这里", counters,
          "可变延迟完成前，关联的 SB counter 保持非零"));
      }
    }
    byId("state-groups").innerHTML = groups.join("");

    var units = ["FP", "INT", "MEM"];
    if (model.usesActive) { units.push("ACTIVE"); }
    byId("unit-bank").innerHTML = units.map(function (unit) {
      var active = unit === "ACTIVE" ? snapshot.resources.active !== null : snapshot.resources.units[unit];
      var title = unit === "ACTIVE"
        ? (active ? "当前 active 指令是 " + snapshot.resources.active : "无全局 active 指令")
        : (active ? unit + " 使用中" : unit + " 可用");
      return hardwareTile(unit, active ? "active" : "available", title,
        evaluation.checked.indexOf(unit) !== -1);
    }).join("");
  }

  function renderVerdict(snapshot) {
    var card = byId("verdict-card");
    var title = byId("verdict-title");
    var reason = byId("verdict-reason");
    var summary = byId("check-summary");
    var inspected = inspectedInstruction(snapshot);
    byId("cycle-number").textContent = String(state.cycle);
    if (inspected === null) {
      card.classList.remove("blocked");
      title.textContent = "DONE";
      reason.textContent = "没有新的候选指令。";
      summary.textContent = "program complete";
      return;
    }
    var evaluation = evaluateInstruction(inspected, snapshot);
    var isCurrent = snapshot.candidate !== null && inspected.id === snapshot.candidate.id;
    var result = evaluation.failed.length > 0 ? "STALL" : (isCurrent ? "ISSUE" : "READY");
    card.classList.toggle("blocked", result === "STALL");
    title.textContent = inspected.id + " · " + result;
    if (inspected.nop) {
      reason.textContent = "编译器插入一个空周期。";
      summary.textContent = "不读取硬件依赖状态";
    }
    else if (result === "STALL") {
      reason.textContent = evaluation.failed.join(", ") + " 使用中。";
      summary.textContent = "橙框表示这条指令读取的状态；其中存在黄色状态";
    }
    else if (result === "READY") {
      reason.textContent = "当前检查可以通过，但程序顺序还没有轮到它。";
      summary.textContent = "橙框状态均可用";
    }
    else {
      reason.textContent = "这条指令读取的状态均可用。";
      summary.textContent = "检查通过";
    }
  }

  function instructionLabel(item) {
    if (state.stage !== "compiler" || item.nop) {
      return item.id + ": " + item.asm;
    }
    var control = item.control;
    return item.id + ": [S=" + control.stall +
      " WB=" + (control.wb || "-") +
      " RD=" + (control.rd || "-") +
      " WAIT={" + control.wait.join(",") + "}] " + item.asm;
  }

  function allCycles(replay) {
    var cycles = [];
    for (var cycle = 0; cycle <= replay.maxCycle; cycle += 1) { cycles.push(cycle); }
    return cycles;
  }

  function displayStatus(status) {
    return { IS: "Is", NP: "nop", EX: "X", WB: "Wb", WAIT: "stl" }[status] || "";
  }

  function renderPipeline(snapshot) {
    var replay = currentReplay();
    var program = currentProgram();
    var cycles = allCycles(replay);
    var html = "<div class=\"pipeline-header label-header\">Instruction</div>";
    cycles.forEach(function (cycle) {
      html += "<div class=\"pipeline-header" + (cycle === state.cycle ? " current-cycle" : "") +
        "\">C" + cycle + "</div>";
    });

    program.forEach(function (item, rowIndex) {
      var candidate = snapshot.candidate && snapshot.candidate.id === item.id;
      var labelClasses = ["instruction-label"];
      if (rowIndex % 2 === 1) { labelClasses.push("row-alt"); }
      if (candidate) {
        labelClasses.push(snapshot.decision === "STALL" ? "blocked-candidate" : "current-candidate");
      }
      var issueCycle = replay.timeline[item.id].findIndex(function (status) {
        return status === "IS" || status === "NP";
      });
      var inspectable = issueCycle === -1 || state.cycle <= issueCycle;
      if (inspectable) { labelClasses.push("inspectable"); }
      html += "<div class=\"" + labelClasses.join(" ") + "\"" +
        (inspectable ? " data-instruction=\"" + item.id + "\"" : "") + ">" +
        "<strong title=\"" + escapeHtml(instructionLabel(item)) + "\">" +
        escapeHtml(instructionLabel(item)) + "</strong></div>";

      cycles.forEach(function (cellCycle) {
        var status = cellCycle <= state.cycle ? replay.timeline[item.id][cellCycle] : "";
        var classes = ["pipeline-cell"];
        if (rowIndex % 2 === 1) { classes.push("row-alt"); }
        if (cellCycle === state.cycle) { classes.push("current-cycle"); }
        if (cellCycle > state.cycle) { classes.push("future"); }
        if (status === "IS") { classes.push("issue"); }
        else if (status === "NP") { classes.push("nop"); }
        else if (status === "EX") { classes.push("execute"); }
        else if (status === "WB") { classes.push("writeback"); }
        else if (status === "WAIT") { classes.push("blocked"); }
        html += "<div class=\"" + classes.join(" ") + "\">" + displayStatus(status) + "</div>";
      });
    });

    byId("pipeline").style.gridTemplateColumns =
      "var(--label-width) repeat(" + cycles.length +
      ", max(var(--cell), calc((100% - var(--label-width)) / 16)))";
    byId("pipeline").innerHTML = html;
    byId("pipeline").querySelectorAll(".instruction-label.inspectable").forEach(function (label) {
      label.addEventListener("mouseenter", function () {
        state.inspectId = label.getAttribute("data-instruction");
        renderHardware(currentSnapshot());
        renderVerdict(currentSnapshot());
      });
      label.addEventListener("mouseleave", function () {
        state.inspectId = null;
        renderHardware(currentSnapshot());
        renderVerdict(currentSnapshot());
      });
    });
    syncPipelineScroll(snapshot, program);
  }

  function syncPipelineScroll(snapshot, program) {
    var scroller = document.querySelector(".pipeline-scroller");
    var firstCycleCell = byId("pipeline").querySelector(".pipeline-header:not(.label-header)");
    var cycleWidth = firstCycleCell === null ? 76 : firstCycleCell.getBoundingClientRect().width;
    var currentIndex = snapshot.candidate === null
      ? program.length - 1
      : Math.max(0, program.indexOf(snapshot.candidate));
    scroller.scrollLeft = Math.max(0, (state.cycle - 10) * cycleWidth);
    scroller.scrollTop = Math.max(0, (currentIndex - 6) * 48);
  }

  function renderControls() {
    var replay = currentReplay();
    byId("previous-button").disabled = state.cycle === 0;
    byId("next-button").disabled = state.cycle === replay.maxCycle;
    byId("play-button").textContent = state.timer === null ? "自动播放" : "暂停";
    byId("cycle-slider").setAttribute("max", String(replay.maxCycle));
    byId("cycle-slider").setAttribute("value", String(state.cycle));
    byId("cycle-max").textContent = String(replay.maxCycle);
  }

  function render() {
    var snapshot = currentSnapshot();
    byId("stage-label").textContent = stageModels[state.stage].label;
    byId("mode-description").textContent = stageModels[state.stage].description;
    renderHardware(snapshot);
    renderVerdict(snapshot);
    renderPipeline(snapshot);
    renderControls();
  }

  function setCycle(nextCycle) {
    state.inspectId = null;
    state.cycle = Math.max(0, Math.min(currentReplay().maxCycle, nextCycle));
    render();
  }

  function stopPlayback() {
    if (state.timer !== null) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
  }

  function togglePlayback() {
    if (state.timer !== null) {
      stopPlayback();
      render();
      return;
    }
    if (state.cycle === currentReplay().maxCycle) { state.cycle = 0; }
    state.timer = window.setInterval(function () {
      if (state.cycle >= currentReplay().maxCycle) {
        stopPlayback();
        render();
        return;
      }
      state.cycle += 1;
      state.inspectId = null;
      render();
    }, 950);
    render();
  }

  function escapeHtml(value) {
    return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/\"/g, "&quot;");
  }

  byId("reset-button").addEventListener("click", function () { stopPlayback(); setCycle(0); });
  byId("previous-button").addEventListener("click", function () { stopPlayback(); setCycle(state.cycle - 1); });
  byId("next-button").addEventListener("click", function () { stopPlayback(); setCycle(state.cycle + 1); });
  byId("play-button").addEventListener("click", togglePlayback);
  byId("cycle-slider").addEventListener("change", function (event) {
    stopPlayback();
    var value = Number(event.currentTarget.value);
    if (Number.isFinite(value)) { setCycle(value); }
  });

  render();
}());
