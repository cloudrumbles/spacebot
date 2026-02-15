import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cx } from "./utils";

export const buttonStyles = cva(
  [
    "cursor-default items-center rounded-xl border font-plex font-semibold tracking-wide outline-none transition-colors duration-100",
    "disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-70",
    "focus:ring-none focus:ring-offset-none cursor-pointer ring-offset-app-box",
  ],
  {
    variants: {
      size: {
        icon: "!p-1",
        lg: "gap-3 px-5 py-2.5 text-sm",
        md: "gap-2.5 px-3.5 py-2 text-sm",
        sm: "gap-2 px-2.5 py-1.5 text-tiny",
        xs: "gap-1.5 px-2 py-1 text-tiny",
      },
      variant: {
        default: [
          "border-accent text-accent shadow-sm",
          "hover:bg-accent/15 hover:text-white",
        ],
        subtle: [
          "border-transparent bg-app-button text-ink shadow-sm",
          "hover:bg-app-hover hover:text-ink",
        ],
        outline: [
          "border-app-line text-ink-dull",
          "hover:border-ink-faint hover:text-ink",
        ],
        dotted: [
          "border-dashed border-app-line text-ink-faint",
          "hover:border-ink-dull hover:text-ink-dull",
        ],
        gray: [
          "border-app-line bg-app-box text-ink shadow-sm",
          "hover:bg-app-hover hover:text-ink",
        ],
        accent: [
          "border-accent/30 bg-accent/10 text-accent shadow-sm",
          "hover:bg-accent/20 hover:text-accent",
        ],
        colored: [
          "border-transparent text-white shadow-sm",
        ],
        bare: "border-transparent bg-transparent text-ink-dull hover:text-ink",
      },
      rounding: {
        none: "",
        left: "rounded-r-none border-r-0",
        right: "rounded-l-none border-l-0",
        both: "",
      },
    },
    defaultVariants: {
      size: "sm",
      variant: "default",
      rounding: "both",
    },
  }
);

export type ButtonBaseProps = VariantProps<typeof buttonStyles>;

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    ButtonBaseProps {
  children?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, rounding, ...props }, ref) => {
    return (
      <button
        className={cx(
          buttonStyles({ variant, size, rounding }),
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";
