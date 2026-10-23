import { describe, expect, it } from "vitest";
import { reconcileChosenModel } from "./modelChoice";

describe("reconcileChosenModel", () => {
  it("behält die Auswahl beim ersten Besuch", () => {
    expect(reconcileChosenModel("a", "b", "")).toEqual({ chosen: "a", seenDefault: "b" });
  });

  it("behält die Auswahl, solange der Standard gleich bleibt", () => {
    expect(reconcileChosenModel("a", "b", "b")).toEqual({ chosen: "a", seenDefault: "b" });
  });

  it("verwirft die Auswahl, wenn der Admin den Standard ändert", () => {
    expect(reconcileChosenModel("a", "c", "b")).toEqual({ chosen: "", seenDefault: "c" });
  });

  it("ändert nichts ohne bekannten Standard", () => {
    expect(reconcileChosenModel("a", "", "b")).toEqual({ chosen: "a", seenDefault: "b" });
  });
});
