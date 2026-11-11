/*
 * Sortable tables, adapted from the WAI-ARIA Authoring Practices Guide
 * sortable table example:
 * https://www.w3.org/WAI/ARIA/apg/patterns/table/examples/sortable-table/
 *
 * Each sortable column header holds a <button>. Clicking it sorts the rows by
 * that column and moves aria-sort to its <th>, so only the sorted column
 * carries it. Cells sort by their data-value attribute, falling back to their
 * text. Columns sort numerically unless their header has data-type="text".
 * Numeric columns sort in descending order on the first click, since a higher
 * score is better, and text columns in ascending order.
 */
(function () {
  "use strict";

  var ICONS = { ascending: "▲", descending: "▼", none: "♢" };

  class SortableTable {
    constructor(tableNode) {
      this.tableNode = tableNode;
      this.tbodyNode = tableNode.querySelector("tbody");
      this.columnHeaders = Array.from(tableNode.querySelectorAll("thead th"));

      this.columnHeaders.forEach((header, columnIndex) => {
        var button = header.querySelector("button");
        if (button) {
          button.addEventListener("click", () => this.handleClick(columnIndex));
        }
      });
    }

    isNumeric(columnIndex) {
      return this.columnHeaders[columnIndex].dataset.type !== "text";
    }

    handleClick(columnIndex) {
      var header = this.columnHeaders[columnIndex];
      var current = header.getAttribute("aria-sort");
      var direction;
      if (current === "ascending") {
        direction = "descending";
      } else if (current === "descending") {
        direction = "ascending";
      } else {
        direction = this.isNumeric(columnIndex) ? "descending" : "ascending";
      }

      this.setColumnHeaderSort(columnIndex, direction);
      this.sortColumn(columnIndex, direction);
    }

    setColumnHeaderSort(columnIndex, direction) {
      this.columnHeaders.forEach((header, index) => {
        var icon = header.querySelector("button span[aria-hidden]");
        if (index === columnIndex) {
          header.setAttribute("aria-sort", direction);
          if (icon) icon.textContent = ICONS[direction];
        } else if (header.querySelector("button")) {
          header.removeAttribute("aria-sort");
          if (icon) icon.textContent = ICONS.none;
        }
      });
    }

    sortColumn(columnIndex, direction) {
      var numeric = this.isNumeric(columnIndex);
      var sign = direction === "ascending" ? 1 : -1;
      var rows = Array.from(this.tbodyNode.rows).map((row, index) => {
        var cell = row.cells[columnIndex];
        var raw = cell.dataset.value !== undefined ? cell.dataset.value : cell.textContent;
        return {
          row: row,
          index: index,
          value: numeric ? parseFloat(raw) : raw.trim().toLowerCase(),
        };
      });

      rows.sort((a, b) => {
        var order = numeric ? a.value - b.value : a.value.localeCompare(b.value);
        return order !== 0 ? sign * order : a.index - b.index;
      });

      rows.forEach((item) => this.tbodyNode.appendChild(item.row));
    }
  }

  function init() {
    document.querySelectorAll("table.sortable").forEach((table) => {
      new SortableTable(table);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
