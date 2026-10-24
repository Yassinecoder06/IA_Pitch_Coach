export interface Landmark {
  x: number; // normalised image coords 0..1
  y: number;
  z?: number;
  visibility?: number;
}

/** Face landmarks needed for gaze (subset of MediaPipe's 478, keyed by original index). */
export interface FaceFrame {
  points: Record<number, Landmark>;
  blendshapes: Record<string, number>;
  /** 4x4 facial transformation matrix, column-major (MediaPipe layout). */
  matrix?: number[];
}

export interface PoseFrame {
  landmarks: Landmark[]; // 33 MediaPipe pose landmarks
}

/** MediaPipe face mesh indices used by the gaze scorer. */
export const FACE_POINTS = {
  rightEyeOuter: 33,
  rightEyeInner: 133,
  rightEyeTop: 159,
  rightEyeBottom: 145,
  leftEyeInner: 362,
  leftEyeOuter: 263,
  leftEyeTop: 386,
  leftEyeBottom: 374,
  rightIris: 468,
  leftIris: 473,
  noseTip: 1,
} as const;

export const FACE_POINT_INDICES = Object.values(FACE_POINTS);

export const GAZE_BLENDSHAPES = [
  'eyeLookInLeft',
  'eyeLookOutLeft',
  'eyeLookUpLeft',
  'eyeLookDownLeft',
  'eyeLookInRight',
  'eyeLookOutRight',
  'eyeLookUpRight',
  'eyeLookDownRight',
  'eyeBlinkLeft',
  'eyeBlinkRight',
];

/** MediaPipe pose landmark indices. */
export const POSE = {
  nose: 0,
  leftEar: 7,
  rightEar: 8,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
} as const;
