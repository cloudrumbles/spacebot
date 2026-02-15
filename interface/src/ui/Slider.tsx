"use client";

import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { cva, type VariantProps } from "class-variance-authority";
import { cx } from "./utils";

const sliderTrackStyles = cva(
  "relative flex w-full grow overflow-hidden rounded-full bg-app-button h-2"
);

const sliderRangeStyles = cva("absolute h-full bg-accent rounded-full");

const sliderThumbStyles = cva(
  [
    "block h-5 w-5 rounded-full border-2 border-accent bg-app-box",
    "ring-offset-app-box transition-colors",
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2",
    "disabled:pointer-events-none disabled:opacity-50",
    "cursor-grab active:cursor-grabbing",
  ]
);

const sliderMarkStyles = cva(
  "absolute w-1 h-1 rounded-full bg-app-line top-1/2 -translate-y-1/2"
);

export interface SliderProps
  extends React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>,
    VariantProps<typeof sliderTrackStyles> {
  marks?: number[];
}

export const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  SliderProps
>(({ className, marks, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cx(
      "relative flex w-full touch-none select-none items-center",
      className
    )}
    {...props}
  >
    <SliderPrimitive.Track className={cx(sliderTrackStyles())}>
      <SliderPrimitive.Range className={cx(sliderRangeStyles())} />
      {marks?.map((mark, index) => (
        <div
          key={index}
          className={cx(sliderMarkStyles())}
          style={{
            left: `${mark}%`,
          }}
        />
      ))}
    </SliderPrimitive.Track>
    {props.defaultValue?.map((_, index) => (
      <SliderPrimitive.Thumb
        key={index}
        className={cx(sliderThumbStyles())}
      />
    )) ||
      (props.value?.map((_, index) => (
        <SliderPrimitive.Thumb
          key={index}
          className={cx(sliderThumbStyles())}
        />
      )) || <SliderPrimitive.Thumb className={cx(sliderThumbStyles())} />)}
  </SliderPrimitive.Root>
));

Slider.displayName = SliderPrimitive.Root.displayName;

export interface SliderLabelProps {
  min: number;
  max: number;
  value: number[];
  className?: string;
}

export const SliderLabel: React.FC<SliderLabelProps> = ({
  min,
  max,
  value,
  className,
}) => (
  <div className={cx("flex items-center justify-between text-xs text-ink-dull", className)}>
    <span>{min}</span>
    <span className="font-medium text-ink">
      {value.length > 1 ? `${value[0]} - ${value[1]}` : value[0]}
    </span>
    <span>{max}</span>
  </div>
);

export interface SliderFieldProps extends SliderProps {
  label?: string;
  description?: string;
  showValue?: boolean;
  minLabel?: string;
  maxLabel?: string;
}

export const SliderField: React.FC<SliderFieldProps> = ({
  label,
  description,
  showValue = true,
  minLabel,
  maxLabel,
  value,
  defaultValue,
  min = 0,
  max = 100,
  className,
  ...props
}) => {
  const currentValue = value || defaultValue || [min];
  
  return (
    <div className={cx("space-y-3", className)}>
      {label && (
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-ink">{label}</span>
          {showValue && (
            <span className="text-sm text-ink-dull">
              {currentValue.length > 1
                ? `${currentValue[0]} - ${currentValue[1]}`
                : currentValue[0]}
            </span>
          )}
        </div>
      )}
      <Slider value={value} defaultValue={defaultValue} min={min} max={max} {...props} />
      {description && <p className="text-xs text-ink-dull">{description}</p>}
      {(minLabel || maxLabel) && (
        <div className="flex items-center justify-between text-xs text-ink-dull">
          <span>{minLabel || min}</span>
          <span>{maxLabel || max}</span>
        </div>
      )}
    </div>
  );
};
